import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmdet3d.registry import MODELS
from .detection3d.target import UNTRACKED_ID
__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, num_groups, N = confidence.shape
    confidence, indices = torch.topk(confidence, k, dim=2)
    # create batch index tensor, (bs, k) to match shape of indices
    batch_indices = torch.arange(
        bs, device=indices.device).view(-1, 1, 1).expand(-1, num_groups, k)

    group_indices = torch.arange(
        num_groups, device=indices.device).view(1, -1, 1).expand(bs, -1, k)
    # check how selected_elements is generated

    outputs = []
    for input_i in inputs:
        selected_elements = input_i[batch_indices, group_indices, indices]  # (bs, num_groups, k, ...)
        outputs.append(selected_elements)
    return confidence, outputs, indices  # Return indices as well


@MODELS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor: int,
        embed_dims: int,
        anchor: str,
        anchor_handler=None,
        num_temp_instances:int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        anchor_grad: bool = True,
        feat_grad:bool = True,
        max_time_interval: float = 2,
        # TRAILAB params
        num_learned_groups: int = 1,
        num_learned_temp_groups: int = 1,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        if anchor_handler is not None:
            anchor_handler = MODELS.build(anchor_handler)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([num_learned_groups, self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.num_learned_groups = num_learned_groups
        self.num_learned_temp_groups = num_learned_temp_groups
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.history_time = None
        self.history_T_global = None
        self.mask = None
        self.cached_confidence = None
        self.cached_indices = None
        self.instance_inds_inference = None
        self.instance_inds_training = None
        self.prev_id = 0

    def get(self, batch_size, timestamp, batched_global2lidar, dn_metas=None):
        if self.training:
            instance_feature = torch.tile(
                self.instance_feature[None], (batch_size, 1, 1, 1)
            )  # (bs, num_groups, num_anchor, embed_dims)
            anchor = torch.tile(
                self.anchor[None], (batch_size, self.num_learned_groups, 1, 1))
        else:
            # only take the first group at inference
            instance_feature = torch.tile(
                self.instance_feature[:1][None], (batch_size, 1, 1, 1)
            )
            anchor = torch.tile(
                self.anchor[None], (batch_size, 1, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
            and instance_feature.shape[1] == self.cached_anchor.shape[1] # instance_feature groups matches cached groups
        ):
            num_groups = instance_feature.shape[1]
            # update time interval and self.mask
            history_time = self.history_time
            time_interval = timestamp - history_time
            time_interval = time_interval.to(
                dtype=instance_feature.dtype, device=instance_feature.device)
            # mask of which instances in the batch are within the max time interval
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                # update all anchors regardless of new sequence
                T_temp2cur = torch.stack(
                    [
                        x @ self.history_T_global[i]
                        for i, x in enumerate(batched_global2lidar)
                    ]
                ).to(self.cached_anchor.device)
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor.flatten(1, 2), # flatten to (bs, num_groups * num_anchor, box size)
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0].reshape(batch_size, num_groups, -1, self.cached_anchor.shape[-1])

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
                # sampler.update_dn handles new sequence case by using instance_bank.mask
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, cls):
        if self.cached_feature is None or self.cached_feature.shape[1]!=instance_feature.shape[1]:
            # no cached instances or different number of groups (training to inference)
            # TODO handle the inference case more elegantly
            return instance_feature, anchor

        num_dn = instance_feature.shape[2] - self.num_anchor
        if num_dn > 0:
            # dn_instance_feature (bs, num_groups, num_dn, embed_dims)
            dn_instance_feature = instance_feature[:, :, -num_dn:, :]
            # anchor (bs, num_groups, num_dn, box size)
            dn_anchor = anchor[:, :, -num_dn:, :]
            instance_feature = instance_feature[:, :, :self.num_anchor, :]
            anchor = anchor[:, :, :self.num_anchor, :]
            cls = cls[:, :, :self.num_anchor, :]

        # take the topk instances with highest confidence
        N = self.num_anchor - self.num_temp_instances
        confidence = cls.max(dim=-1).values
        _, (selected_feature, selected_anchor), _ = topk(
            confidence, N, instance_feature, anchor
        )
        # concatenate with cached queries (TQ)
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=2
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=2
        )
        # mask determines which items in the batch should be updated with selected_feature.
        # otherwise, if mask is False, the item should be updated with the original feature.
        instance_feature = torch.where(
            self.mask[:, None, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(
            self.mask[:, None, None, None], selected_anchor, anchor)

        # update instance_inds with new instances
        if self.instance_inds_inference is not None:
            # wipe the stored memory based on self.mask (determined by difference in timestamp)
            self.instance_inds_inference = torch.where(
                self.mask[:, None, None],
                self.instance_inds_inference,
                self.instance_inds_inference.new_tensor(UNTRACKED_ID),
            )

        if num_dn > 0:  # add back dn queries
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=2
            )
            anchor = torch.cat([anchor, dn_anchor], dim=2)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_preds: torch.Tensor,
        timestamp,
        batch_history_T_global,
        instance_inds=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        cls_preds = cls_preds.detach()

        self.history_time = timestamp
        self.history_T_global = batch_history_T_global
        confidence = cls_preds.max(dim=-1).values.sigmoid()
        if self.cached_confidence is not None:
            # update temp instances confidence with decay
            confidence[:, :, : self.num_temp_instances] = torch.maximum(
                self.cached_confidence * self.confidence_decay, # cached confidence * decay
                confidence[:, :, : self.num_temp_instances], # current confidence
            )

        # self.cached_confidence used to perform confidence decay in the next step
        (
            self.cached_confidence,
            (self.cached_feature, self.cached_anchor),
            self.cached_indices,
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
        if self.num_temp_instances > 0 and instance_inds is not None:
            # update and cache instance_inds for the next frame
            # only used at inference time
            self.update_instance_inds(instance_inds, self.cached_indices)

    def get_instance_ind(self, cls_pred: torch.Tensor, threshold:float = 0.0):
        # convert class prediction to confidence
        confidence = cls_pred.max(dim=-1).values.sigmoid()
        # initialize empty instance_inds
        instance_inds = confidence.new_full(confidence.shape, UNTRACKED_ID, dtype=torch.long)
        if (
            self.instance_inds_inference is not None  # not first frame of sequence
            and self.instance_inds_inference.shape[0] == instance_inds.shape[0]
        ):
            # expect both past inds and new inds to have the same shape
            assert self.instance_inds_inference.shape[2] == instance_inds.shape[2], (
                self.instance_inds_inference.shape,
                instance_inds.shape,
            ) # sanity check
            # assign instance_ids from the past frame
            instance_inds[:, :, :self.instance_inds_inference.shape[2]
                          ] = self.instance_inds_inference
        # for instances with no ID including PQ and untracked TQ
        mask = instance_inds == UNTRACKED_ID
        # for instances with confidence above threshold
        mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum().item()
        # assign them new IDs
        new_ids = torch.arange(
            self.prev_id, 
            self.prev_id + num_new_instance, 
            dtype=torch.long,
            device=instance_inds.device)
        # assign new IDs across the entire batch
        instance_inds[torch.where(mask)] = new_ids
        # update prev_id
        self.prev_id += num_new_instance
        return instance_inds

    def update_instance_inds(self, instance_inds, topk_indices):
        """
        Prepare self.instance_inds for the next frame, appending 300 new instances of value -1 to the end (for the PQ)
        Only used at inference time.
        """
        bs, num_groups, k = topk_indices.shape
        batch_indices = torch.arange(bs).view(
            -1, 1, 1).expand(-1, num_groups, k)
        group_indices = torch.arange(num_groups).view(
            1, -1, 1).expand(bs, -1, k)
        instance_inds = instance_inds[batch_indices, group_indices, topk_indices]
        # pad with -1 on the end
        self.instance_inds_inference = F.pad(
            instance_inds,
            (0, self.num_anchor - self.num_temp_instances),
            value=UNTRACKED_ID,
        )
