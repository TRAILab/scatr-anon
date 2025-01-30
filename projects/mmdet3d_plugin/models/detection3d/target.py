from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from scipy.optimize import linear_sum_assignment

from projects.mmdet3d_plugin.core.box3d import *

from ..base_target import BaseTargetWithDenoising

__all__ = ["SparseBox3DTarget"]

UNTRACKED_ID = -1
PAD_CLS_TARGET = -1
NEG_DN_CLS_TARGET = -3

@MODELS.register_module()
class SparseBox3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=[1.0] * 8 + [0.0] * 2,
        cls_wise_reg_weights={},
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        supervise_qc: bool = False,
    ):
        super(SparseBox3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn
        self.supervise_qc = supervise_qc  # supervise query consistency

    def encode_reg_target(self, box_target: List[LiDARInstance3DBoxes], device=None):
        outputs = []
        for box in box_target:
            output = self.encode_reg_target_single(box, device)
            outputs.append(output)
        return outputs

    def encode_reg_target_single(self, box_target_i: LiDARInstance3DBoxes, device=None):
        output = torch.cat(
            [
                box_target_i.gravity_center,
                box_target_i.dims.log(),
                torch.sin(box_target_i.yaw).unsqueeze(-1),
                torch.cos(box_target_i.yaw).unsqueeze(-1),
                box_target_i.tensor[:, YAW+1:],  # velocity vector
            ],
            dim=-1,
        )
        if device is not None:
            output = output.to(device=device)
        return output

    def sample(
            self,
            cls_pred,
            box_pred,
            cls_gt,
            box_gt,
            id_gt,
            prev_inst_inds=None,
    ):
        bs, num_pred, num_cls = cls_pred.shape
        if prev_inst_inds is None:
            prev_inst_inds = [None] * bs
        cls_target = []
        box_target = []
        reg_weights = []
        id_target = []
        cls_pred_act = cls_pred.detach().sigmoid()
        for batch_idx, (cls_pred_act_i, box_pred_i, cls_gt_i, box_gt_i, id_gt_i, prev_inst_inds_i) in enumerate(zip(
            cls_pred_act, box_pred, cls_gt, box_gt, id_gt, prev_inst_inds
        )):
            cls_target_i, box_target_i, reg_weights_i, id_target_i = self.sample_single(
                cls_pred_act_i.detach(),
                box_pred_i.detach(),
                cls_gt_i,
                box_gt_i,
                id_gt_i,
                prev_inst_inds_i,
            )
            cls_target.append(cls_target_i)
            box_target.append(box_target_i)
            reg_weights.append(reg_weights_i)
            id_target.append(id_target_i)
        # stack to tensor along batch dimension
        cls_target = torch.stack(cls_target)
        box_target = torch.stack(box_target)
        reg_weights = torch.stack(reg_weights)
        id_target = torch.stack(id_target)
        return cls_target, box_target, reg_weights, id_target

    def sample_single(
            self,
            cls_pred_act_i,
            box_pred_i,
            cls_gt_i,
            box_gt_i,
            id_gt_i,
            prev_inst_inds_i=None,):
        """
        Sample targets for predictions in a single item from the batch.
        Not batched.
        # TODO: Replace match cost and assignment code with code from the latest mmdet.
        # The current implementation uses a custom matching cost and assignment logic.
        # The new implementation should leverage the latest mmdet library's utilities
        # for computing matching costs and performing the assignment, which are expected
        # to be more efficient and cleaner. This involves:
        # 1. Importing the necessary functions from mmdet.
        # 2. Replacing the custom cost computation with mmdet's cost computation.
        # 3. Using mmdet's assignment function to replace the current linear_sum_assignment.
        # The current implementation uses a custom method for computing the match cost and performing the assignment.
        # The latest mmdet library has a more optimized and cleaner implementation for these operations.
        # Refer to the mmdet3d/models/detectors/assigners/ directory in the mmdet repository for the latest code.
        # Specifically, look at the HungarianAssigner3D class and its methods for computing the cost and performing the assignment.
        """
        # construct targets
        num_preds, num_cls = cls_pred_act_i.shape
        cls_target_i = cls_pred_act_i.new_full(
            (num_preds,), num_cls, dtype=torch.long)
        box_target_i = box_pred_i.new_zeros(box_pred_i.shape)
        reg_weights_i = box_pred_i.new_zeros(box_pred_i.shape)
        id_target_i = box_pred_i.new_full((num_preds,), UNTRACKED_ID, dtype=torch.long)
        track_id_2_gt_ind = {track_id.item():gt_ind for gt_ind, track_id in enumerate(id_gt_i)}

        # in the case of no gt objects to assign
        if len(cls_gt_i) == 0:
            return cls_target_i, box_target_i, reg_weights_i, id_target_i

        encoded_box_gt_i = self.encode_reg_target_single(
            box_gt_i, box_pred_i.device)

        # if encoded is nan, set reg_weights to 0
        instance_reg_weights_i = torch.logical_not(
            encoded_box_gt_i.isnan()).to(dtype=encoded_box_gt_i.dtype)
        # set reg_weights by class
        # used to ignore orientation for traffic cones
        for class_label, weight in self.cls_wise_reg_weights.items():
            instance_reg_weights_i = torch.where(
                (cls_gt_i == class_label)[:, None],
                instance_reg_weights_i.new_tensor(weight),
                instance_reg_weights_i,
            )

        # split predictions between tq and non-tq
        # mask on preds
        tq_mask = cls_gt_i.new_zeros((num_preds,), dtype=torch.bool)
        if prev_inst_inds_i is not None and self.supervise_qc:
            # mask on gt
            nb_obj_mask = torch.logical_not(torch.isin(id_gt_i, prev_inst_inds_i))
            num_tq = len(prev_inst_inds_i)
            tq_mask[:num_tq] = prev_inst_inds_i != UNTRACKED_ID
            pq_mask = torch.logical_not(tq_mask)
            # construct target for prev tracked objects
            for pred_idx, prev_inst_ind in enumerate(prev_inst_inds_i):
                if prev_inst_ind == UNTRACKED_ID:
                    continue
                gt_ind = track_id_2_gt_ind.get(prev_inst_ind.item(), None)
                if gt_ind is None:
                    continue
                cls_target_i[pred_idx] = cls_gt_i[gt_ind]
                box_target_i[pred_idx] = encoded_box_gt_i[gt_ind]
                reg_weights_i[pred_idx] = instance_reg_weights_i[gt_ind]
                id_target_i[pred_idx] = id_gt_i[gt_ind]
            cls_pred_act_pq = cls_pred_act_i[pq_mask]
            box_pred_pq = box_pred_i[pq_mask]
            cls_gt_nb = cls_gt_i[nb_obj_mask]
            encoded_box_gt_nb = encoded_box_gt_i[nb_obj_mask]
            id_gt_nb = id_gt_i[nb_obj_mask]
            instance_reg_weights_nb = instance_reg_weights_i[nb_obj_mask]
        else:
            pq_mask = torch.logical_not(tq_mask)
            num_tq = 0
            cls_pred_act_pq = cls_pred_act_i
            box_pred_pq = box_pred_i
            cls_gt_nb = cls_gt_i
            encoded_box_gt_nb = encoded_box_gt_i
            id_gt_nb = id_gt_i
            instance_reg_weights_nb = instance_reg_weights_i
        # perform hungarian assignment on remaining predictions
        if len(cls_gt_nb) != 0:
            cls_target_pq, box_target_pq, reg_weights_pq, id_target_pq = self._sample_single_pq(
                cls_pred_act_pq,
                box_pred_pq,
                cls_gt_nb,
                encoded_box_gt_nb,
                id_gt_nb,
                instance_reg_weights_nb,
            )

            # merge pq targets back into total target
            cls_target_i[pq_mask] = cls_target_pq
            box_target_i[pq_mask] = box_target_pq
            reg_weights_i[pq_mask] = reg_weights_pq
            id_target_i[pq_mask] = id_target_pq

        return cls_target_i, box_target_i, reg_weights_i, id_target_i

    def _sample_single_pq(
            self,
            cls_pred_act_i,
            box_pred_i,
            cls_gt_i,
            encoded_box_gt_i,
            id_gt_i,
            instance_reg_weights_i
    ):
        num_preds, num_cls = cls_pred_act_i.shape
        cls_target_i = cls_pred_act_i.new_full(
            (num_preds,), num_cls, dtype=torch.long)
        box_target_i = box_pred_i.new_zeros(box_pred_i.shape)
        reg_weights_i = box_pred_i.new_zeros(box_pred_i.shape)
        id_target_i = box_pred_i.new_full((num_preds,), UNTRACKED_ID, dtype=torch.long)

        # in the case of no gt objects to assign
        if len(cls_gt_i) == 0:
            return cls_target_i, box_target_i, reg_weights_i, id_target_i

        # compute assignment costs
        cls_cost_i = self._cls_cost_single(cls_pred_act_i, cls_gt_i)
        box_cost_i = self._box_cost_single(
            box_pred_i, encoded_box_gt_i, instance_reg_weights_i)

        # perform hungarian matching based on costs
        cost = (cls_cost_i + box_cost_i).detach().cpu().numpy()
        cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
        pred_idx, target_idx = linear_sum_assignment(cost)
        pred_idx = torch.from_numpy(pred_idx).to(dtype=torch.long)
        target_idx = torch.from_numpy(target_idx).to(dtype=torch.long)

        # insert gt based on assigned indices
        cls_target_i[pred_idx] = cls_gt_i[target_idx]
        box_target_i[pred_idx] = encoded_box_gt_i[target_idx]
        reg_weights_i[pred_idx] = instance_reg_weights_i[target_idx]
        id_target_i[pred_idx] = id_gt_i[target_idx]
        return cls_target_i, box_target_i, reg_weights_i, id_target_i

    def _cls_cost_single(self, cls_pred_act_i, cls_target_i):
        """
        Compute the class cost between the predicted and target classes.
        Follows the focal loss formulation.
        cls_pred_act is activated (sigmoid applied, range [0, 1])
        """
        eps = torch.finfo(cls_pred_act_i.dtype).eps
        neg_cost = (
            -(1 - cls_pred_act_i + eps).log()
            * (1 - self.alpha)
            * cls_pred_act_i.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred_act_i + eps).log()
            * self.alpha
            * (1 - cls_pred_act_i).pow(self.gamma)
        )
        return (
            (pos_cost[:, cls_target_i] - neg_cost[:, cls_target_i])
            * self.cls_weight
        )

    def _box_cost_single(self, box_pred_i, box_target_i, instance_reg_weights_i):
        """
        Compute the box cost between the predicted and target boxes.
        """
        return torch.sum(
            torch.abs(box_pred_i[:, None] - box_target_i[None])
            * instance_reg_weights_i[None]
            * box_pred_i.new_tensor(self.reg_weights),
            dim=-1,
        ) * self.box_weight

    def get_dn_anchors(self, cls_target, box_target, gt_instance_inds=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_inds = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_inds is not None:
                gt_instance_inds = [x[: self.max_dn_gt]
                                    for x in gt_instance_inds]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        # pad to max_dn_gt
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=PAD_CLS_TARGET)
                for x in cls_target
            ]
        )
        box_target = self.encode_reg_target(box_target, cls_target.device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )
        box_target = torch.where(
            cls_target[..., None] == PAD_CLS_TARGET, box_target.new_tensor(0), box_target
        )
        if gt_instance_inds is not None:
            gt_instance_inds = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_inds
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_inds is not None:
                gt_instance_inds = gt_instance_inds.tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = cls_target.new_full(cls_target.shape, NEG_DN_CLS_TARGET)
        if gt_instance_inds is not None:
            dn_id_target = gt_instance_inds.new_full(
                gt_instance_inds.shape, UNTRACKED_ID
            )
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_inds is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            box_cost_i = self._box_cost_single(
                dn_anchor[i], box_target[i], torch.ones_like(box_target[i])
            )
            cost = box_cost_i.cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_inds is not None:
                dn_id_target[i, anchor_idx] = gt_instance_inds[i, gt_idx]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_inds is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        # valid denotes dn queries corresponding to a gt
        valid_mask = (dn_cls_target != PAD_CLS_TARGET) & (dn_cls_target != NEG_DN_CLS_TARGET)
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            # valid mask denotes dn queries corresponding to a gt or
            # negative dn queries
            valid_mask = torch.logical_or(
                valid_mask,
                (cls_target != PAD_CLS_TARGET) & (dn_cls_target == NEG_DN_CLS_TARGET)
            )
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_normal_anchor,
        temporal_valid_mask,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_normal_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_normal_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_normal_anchor]
        anchor = anchor[:, :num_normal_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_inds
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        # pad the temp_dn_metas to the same length of dn_metas
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        if self.num_temp_dn_groups <= 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = (
            torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        )
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            )[:, temp_group_mask]
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )
