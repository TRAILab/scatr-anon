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

# tracking ID of untracked/unassigned query
UNTRACKED_ID = -1
# CLS target of DN query, used to pad groups
PAD_CLS_TARGET = -1
# CLS target of negative/unassigned DN query
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
        self.reg_weights = torch.tensor(reg_weights)
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = torch.tensor(dn_noise_scale)
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
        bs, num_learned_groups, num_pred, num_cls = cls_pred.shape
        if prev_inst_inds is None:
            prev_inst_inds = [None] * bs
        cls_target = []
        box_target = []
        reg_weights = []
        id_target = []
        cls_pred_act = cls_pred.detach().sigmoid()
        for batch_idx, (cls_pred_act_i, box_pred_i, cls_gt_i, box_gt_i, id_gt_i, prev_inst_inds_i) in enumerate(zip(
            cls_pred_act, box_pred, cls_gt, box_gt, id_gt, prev_inst_inds
        )): # construct pred targets for each sample in batch
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
        num_groups, num_preds, num_cls = cls_pred_act_i.shape
        cls_target_i = cls_pred_act_i.new_full((num_groups, num_preds,), num_cls, dtype=torch.long)  # (num groups, num_preds)
        box_target_i = torch.zeros_like(box_pred_i)
        reg_weights_i = torch.zeros_like(box_pred_i)
        id_target_i = id_gt_i.new_full((num_groups, num_preds,), UNTRACKED_ID, dtype=torch.long)

        # in the case of no gt objects to assign
        if len(cls_gt_i) == 0:
            return cls_target_i, box_target_i, reg_weights_i, id_target_i

        # encode bbox, normalize wlh, split yaw into sin/cos
        encoded_box_gt_i = self.encode_reg_target_single(
            box_gt_i, box_pred_i.device)

        # if encoded is nan, set reg_weights to 0, else 1
        # (num gt, bbox size)
        instance_reg_weights_i = torch.logical_not(
            encoded_box_gt_i.isnan()).to(dtype=encoded_box_gt_i.dtype)

        # set reg_weights by class
        # used to ignore orientation for traffic cones
        for class_label, weight in self.cls_wise_reg_weights.items():
            # (num gt, bbox size)
            instance_reg_weights_i = torch.where(
                (cls_gt_i == class_label)[:, None],  # (num gt, 1)
                instance_reg_weights_i.new_tensor(weight),
                instance_reg_weights_i,
            )

        # split predictions between tq and non-tq
        # mask on preds
        tq_mask = cls_gt_i.new_zeros((num_groups, num_preds,), dtype=torch.bool)
        if prev_inst_inds_i is not None and self.supervise_qc:
            track_id_2_gt_ind = {
                track_id.item(): gt_ind for gt_ind, track_id in enumerate(id_gt_i)}
            # mask on gt
            nb_obj_mask = torch.logical_not(torch.isin(id_gt_i, prev_inst_inds_i))
            _, num_tq = prev_inst_inds_i.shape
            tq_mask[:, :num_tq] = prev_inst_inds_i != UNTRACKED_ID

            # construct target for prev tracked objects
            # prev_inds_valid_mask corresponds to tq that were not UNTRACKED in the prev frame
            prev_inds_valid_mask = prev_inst_inds_i != UNTRACKED_ID  # (num_groups, num_tq)
            for group_idx in range(num_groups):
                # tensor of prev_inst_inds that are not UNTRACKED_ID
                valid_prev_inst_inds = prev_inst_inds_i[group_idx][prev_inds_valid_mask[group_idx]]
                # indices of the gt objects in curr frame that were also in the prev frame.
                # if the prev inst ind is not in the track_id_2_gt_ind (i.e. not in the current frame), set to -1
                gt_inds = torch.tensor([
                    track_id_2_gt_ind.get(track_id.item(), -1) for track_id in valid_prev_inst_inds]) 

                # Filter valid gt indices
                valid_gt_mask = gt_inds != -1  # (num gt obj from prev iteration)
                valid_gt_inds = gt_inds[valid_gt_mask] # (num tracked/non-nb gt in current frame)
                # and corresponding valid pred indices. (num tq) --> nonzero, (num gt obj from prev iter) --> valid_gt_mask (num_tracked/non-nb gt)
                valid_pred_inds = torch.nonzero(prev_inds_valid_mask[group_idx], as_tuple=False).squeeze(1)[valid_gt_mask] # (num tracked/non-nb gt)
                # construct target for tq
                cls_target_i[group_idx, valid_pred_inds] = cls_gt_i[valid_gt_inds]
                box_target_i[group_idx, valid_pred_inds] = encoded_box_gt_i[valid_gt_inds]
                reg_weights_i[group_idx, valid_pred_inds] = instance_reg_weights_i[valid_gt_inds]
                id_target_i[group_idx, valid_pred_inds] = id_gt_i[valid_gt_inds]

            # pq can include unassigned tq
            pq_mask = torch.logical_not(tq_mask)  # (num groups, num_preds)
            # pq preds
            cls_pred_act_pq = [cls_pred_act[mask_i] for cls_pred_act, mask_i in zip(cls_pred_act_i, pq_mask)]
            box_pred_pq = [box_pred[mask_i] for box_pred, mask_i in zip(box_pred_i, pq_mask)]
            # newborn labels
            cls_gt_nb = cls_gt_i[nb_obj_mask]
            encoded_box_gt_nb = encoded_box_gt_i[nb_obj_mask]
            id_gt_nb = id_gt_i[nb_obj_mask]
            instance_reg_weights_nb = instance_reg_weights_i[nb_obj_mask]
        else:
            pq_mask = torch.logical_not(tq_mask)  # (num groups, num_preds)
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
            for group_idx in range(num_groups):
                cls_target_i[group_idx, pq_mask[group_idx]] = cls_target_pq[group_idx]
                box_target_i[group_idx, pq_mask[group_idx]] = box_target_pq[group_idx]
                reg_weights_i[group_idx, pq_mask[group_idx]] = reg_weights_pq[group_idx]
                id_target_i[group_idx, pq_mask[group_idx]] = id_target_pq[group_idx].to(torch.long)

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
        """
        Generate targets for proposal query predictions in a single item from the batch.
        The pq preds are grouped.
        Each group may have a different number of preds
        """
        num_groups = len(cls_pred_act_i)
        num_cls = cls_pred_act_i[0].shape[-1]
        # use num_cls as cls_id of background assignment
        # cls_target_i (num_preds, 1)
        cls_target_i = [torch.full_like(cls_pred[..., 0], num_cls, dtype=torch.long) for cls_pred in cls_pred_act_i]
        box_target_i = [torch.zeros_like(box_pred) for box_pred in box_pred_i]
        reg_weights_i = [torch.zeros_like(box_pred) for box_pred in box_pred_i]
        id_target_i = [id_gt_i.new_full((cls_pred.shape[0],), UNTRACKED_ID, dtype=torch.long) for cls_pred in cls_pred_act_i]

        # in the case of no gt objects to assign
        if len(cls_gt_i) == 0:
            return cls_target_i, box_target_i, reg_weights_i, id_target_i

        # compute assignment costs for all groups
        cls_cost = [self._cls_cost_group(cls_pred, cls_gt_i) for cls_pred in cls_pred_act_i]
        box_cost = [
            self._box_cost_group(box_pred, encoded_box_gt_i, instance_reg_weights_i)
            for box_pred in box_pred_i]

        # perform hungarian matching based on costs
        cost = [(cls_cost[group_idx] + box_cost[group_idx]).detach().cpu().numpy() for group_idx in range(num_groups)]
        cost = [np.where(np.isneginf(c) | np.isnan(c), np.inf, c) for c in cost]
        pred_idx, target_idx = zip(*[linear_sum_assignment(c) for c in cost])
        pred_idx = [torch.from_numpy(idx) for idx in pred_idx]
        target_idx = [torch.from_numpy(idx) for idx in target_idx]

        # insert gt based on assigned indices
        for group_idx in range(num_groups):
            cls_target_i[group_idx][pred_idx[group_idx]] = cls_gt_i[target_idx[group_idx]]
            box_target_i[group_idx][pred_idx[group_idx]] = encoded_box_gt_i[target_idx[group_idx]]
            reg_weights_i[group_idx][pred_idx[group_idx]] = instance_reg_weights_i[target_idx[group_idx]]
            id_target_i[group_idx][pred_idx[group_idx]] = id_gt_i[target_idx[group_idx]]
        return cls_target_i, box_target_i, reg_weights_i, id_target_i

    def _cls_cost_group(self, cls_pred_act_i, cls_target_i):
        """
        Compute the class cost between the predicted and target classes for a group of predictions.
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
            (pos_cost[..., cls_target_i] - neg_cost[..., cls_target_i])
            * self.cls_weight
        )

    def _box_cost_group(self, box_pred_i, box_target_i, instance_reg_weights_i):
        """
        Compute the box cost between the predicted and target boxes for a group of predictions.
        box_pred_i: (num_groups, num_preds, box_dim)
        box_target_i: (num_gt, box_dim)
        """
        if len(box_pred_i.shape) == 3:
            return torch.sum(
                torch.abs(box_pred_i[:, :, None] - box_target_i[None, None])
                * instance_reg_weights_i[None, None]
                * self.reg_weights.to(box_pred_i.device),
                dim=-1,
            ) * self.box_weight
        else:
            assert len(box_pred_i.shape) == 2
            return torch.sum(
                torch.abs(box_pred_i[:, None] - box_target_i[None])
                * instance_reg_weights_i[None]
                * self.reg_weights.to(box_pred_i.device),
                dim=-1,
            ) * self.box_weight

    def get_dn_anchors(self, cls_target, box_target, gt_instance_inds=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_inds = None

        trimmed_cls_target = []
        trimmed_box_target = []
        trimmed_gt_instance_inds = [] if gt_instance_inds is not None else None
        for batch_idx, (cls_target_i, box_target_i) in enumerate(zip(cls_target, box_target)):
            if len(cls_target_i) > self.max_dn_gt:
                # randomly select self.max_dn_gt gt boxes for each element in the batch
                rand_indices = torch.randperm(len(cls_target_i))[
                    : self.max_dn_gt]
                trimmed_cls_target.append(cls_target_i[rand_indices])
                trimmed_box_target.append(box_target_i[rand_indices])
                if gt_instance_inds is not None:
                    trimmed_gt_instance_inds.append(
                        gt_instance_inds[batch_idx][rand_indices])
            else:
                trimmed_cls_target.append(cls_target_i)
                trimmed_box_target.append(box_target_i)
                if gt_instance_inds is not None:
                    trimmed_gt_instance_inds.append(gt_instance_inds[batch_idx])
        cls_target = trimmed_cls_target
        box_target = trimmed_box_target
        gt_instance_inds = trimmed_gt_instance_inds

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        # TODO add extra negative queries instead of padding with blank queries
        # pad targets to max_dn_gt, the largets num gt across the batch
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
        )  # (bs, num_gt_padded, box_dim)
        if gt_instance_inds is not None:
            gt_instance_inds = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=UNTRACKED_ID)
                    for x in gt_instance_inds
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            # (bs, num_dn_groups, max_num_gt)
            cls_target = cls_target.unsqueeze(1).tile(1, self.num_dn_groups, 1)
            # (bs, num_dn_groups, max_num_gt, box_dim)
            box_target = box_target.unsqueeze(1).tile(1, self.num_dn_groups, 1, 1)
            if gt_instance_inds is not None:
                gt_instance_inds = gt_instance_inds.unsqueeze(1).tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= self.dn_noise_scale.to(noise.device)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= self.dn_noise_scale.to(noise_neg.device)
            # (bs, num_dn_groups, num_gt * 2, box_dim)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=2)
            num_gt *= 2

        # (bs, num_dn_groups, num_gt * 2 if add_neg_dn, box_dim)
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = cls_target.new_full(cls_target.shape, NEG_DN_CLS_TARGET)
        if gt_instance_inds is not None:
            dn_id_target = gt_instance_inds.new_full(
                gt_instance_inds.shape, UNTRACKED_ID
            )
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=2)
            if gt_instance_inds is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=2)

        for batch_idx in range(bs):  
            # TODO compute a batched cost matrix for efficiency
            # box_target and cls_target are tiled from (bs, num_gt) to (bs, num_dn_groups, num_gt)
            box_cost_i = self._box_cost_group(
                dn_anchor[batch_idx], box_target[batch_idx, 0], torch.ones_like(box_target[batch_idx, 0])
            ) # (num_dn_groups, num dn queries, num_gt)
            cost = box_cost_i.cpu().numpy()
            for group_idx, cost_i in enumerate(cost):
                anchor_idx, gt_idx = linear_sum_assignment(cost_i)
                anchor_idx = torch.from_numpy(anchor_idx)
                gt_idx = torch.from_numpy(gt_idx)
                dn_box_target[batch_idx, group_idx, anchor_idx] = box_target[batch_idx, 0, gt_idx]
                dn_cls_target[batch_idx, group_idx, anchor_idx] = cls_target[batch_idx, 0, gt_idx]
                if gt_instance_inds is not None:
                    dn_id_target[batch_idx, group_idx, anchor_idx] = gt_instance_inds[batch_idx, 0, gt_idx]

        # valid denotes dn queries corresponding to a gt
        valid_mask = (dn_cls_target != PAD_CLS_TARGET) & (dn_cls_target != NEG_DN_CLS_TARGET)
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
            )
            # valid mask denotes dn queries corresponding to a gt or
            # negative dn queries
            valid_mask = torch.logical_or(
                valid_mask,
                (cls_target != PAD_CLS_TARGET) & (dn_cls_target == NEG_DN_CLS_TARGET)
            )
        # Construct attn_mask, attend within the same dn group
        attn_mask = torch.block_diag(*[torch.ones((num_gt, num_gt), dtype=torch.bool) for _ in range(self.num_dn_groups)])
        attn_mask = attn_mask.to(dn_box_target.device)
        attn_mask = ~attn_mask  # invert mask to make False mean attend

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
        """
        Update DN groups with cached dn group from previous timestep
        """
        bs, num_learned_groups, num_anchor = instance_feature.shape[:3]
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
        # (bs, num learned grps, num queries, ...)
        num_dn_per_group = self.max_dn_gt if not self.add_neg_dn else self.max_dn_gt * 2
        num_dn = num_dn_per_group * (self.num_dn_groups // num_learned_groups)
        # sanity check, total num non-learned queries
        assert num_dn == num_anchor - num_normal_anchor, f"num_dn: {num_dn}, num_anchor: {num_anchor}, num_normal_anchor: {num_normal_anchor}"
        # dn queries
        # (bs, num learned grps, num_dn, ...)
        dn_feat = instance_feature[:, :, -num_dn:]
        dn_anchor = anchor[:, :, -num_dn:]
        # learned queries
        instance_feature = instance_feature[:, :, :num_normal_anchor]
        anchor = anchor[:, :, :num_normal_anchor]

        # not already reshaped with dn_group dim
        if dn_feat.shape[1] != self.num_dn_groups:
            # this case is when there are multiple dn groups per learned group
            raise NotImplementedError(
                "num_learned_groups should be the same as the number of dn groups")
            # TODO test the code below
            # reshape from (bs, num learned groups, num_dn_per_group * num dn groups per learned group) to
            # (bs, total num dn groups, num_dn_per_group, ...)
            # reshape all dn metas from (bs,num_all_dn,xxx)
            # to (bs, dn_group, num_dn_per_group, xxx)
            dn_feat = dn_instance_feature.reshape(
                bs, self.num_dn_groups, num_dn_per_group, -1)
            dn_anchor = dn_anchor.reshape(
                bs, self.num_dn_groups, num_dn_per_group, -1)
            dn_reg_target = dn_reg_target.reshape(
                bs, self.num_dn_groups, num_dn_per_group, -1)
            dn_cls_target = dn_cls_target.reshape(
                bs, self.num_dn_groups, num_dn_per_group)
            valid_mask = valid_mask.reshape(
                bs, self.num_dn_groups, num_dn_per_group)
            if dn_id_target is not None:
                dn_id = dn_id_target.reshape(
                    bs, self.num_dn_groups, num_dn_per_group)

        # update temp_dn_metas by instance_inds
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        # sanity check, they should be the same
        assert num_temp_dn == num_dn, f"num_temp_dn: {num_temp_dn}, num_dn: {num_dn}"
        temp_dn_id = self.dn_metas["dn_id_target"]

        # match represents a matrix of matched ids from the temp dn and current dm
        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id_target[:, :num_temp_dn_groups, None]
        # TODO switch from match to using torch.isin. Test for correctness.        
        # if no matches, set the temp_reg_target to 0
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        # if no matches, set the temp_cls_target to NEG_DN_CLS_TARGET
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(NEG_DN_CLS_TARGET),
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
            dn_id_target,
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
            # update temp_meta if valid_mask is True (is it from the same seq)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            # cat along the num_group dim
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            output.append(meta)
        # append dn features and dn anchors to learned queries and anchors
        output[0] = torch.cat([instance_feature, output[0]], dim=2)
        output[1] = torch.cat([anchor, output[1]], dim=2)
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
        temp_group_mask = (
            torch.randperm(self.num_dn_groups) < self.num_temp_dn_groups
        )
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature.detach()[
                :, temp_group_mask],
            dn_anchor=dn_anchor.detach()[:, temp_group_mask],
            dn_cls_target=dn_cls_target[:, temp_group_mask],
            valid_mask=valid_mask[:, temp_group_mask],
        )
        if dn_id_target is not None:
            self.dn_metas["dn_id_target"] = dn_id_target[:, temp_group_mask]
        else:
            self.dn_metas["dn_id_target"] = None
