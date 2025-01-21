import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmdet3d.registry import MODELS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    # create batch index tensor, (bs, k) to match shape of indices
    batch_indices = torch.arange(
        bs, device=indices.device).unsqueeze(-1).expand(-1, k)

    outputs = []
    for input in inputs:
        selected_elements = input[batch_indices, indices]  # (bs, k, ...)
        outputs.append(selected_elements)
    return confidence, outputs, indices  # Return indices as well


@MODELS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
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
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        # self.metas = None
        self.history_time = None
        self.history_T_global = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.cached_indices = None
        self.instance_inds = None
        self.prev_id = 0

    def get(self, batch_size, timestamp, batched_global2lidar, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            # history_time = self.metas["timestamp"]
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
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

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

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        # take the topk instances with highest confidence
        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor), _ = topk(
            confidence, N, instance_feature, anchor
        )
        # concatenate with cached queries (TQ)
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        # mask determines which items in the batch should be updated with selected_feature.
        # otherwise, if mask is False, the item should be updated with the original feature.
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        # update instance_inds with new instances
        if self.instance_inds is not None:
            # wipe the stored memory based on self.mask (determined by difference in timestamp)
            self.instance_inds = torch.where(
                self.mask[:, None],
                self.instance_inds,
                self.instance_inds.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        timestamp,
        batch_history_T_global,
        instance_inds=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        # self.metas = metas
        self.history_time = timestamp
        self.history_T_global = batch_history_T_global
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            # update confidence with decay
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
            self.cached_indices,
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
        if self.num_temp_instances > 0 and instance_inds is not None:
            # cache instance_inds for the next frame
            self.update_instance_inds(
                instance_inds, confidence, self.cached_indices)

    def get_instance_ind(self, confidence, anchor=None, threshold=None):
        # convert class prediction to confidence
        confidence = confidence.max(dim=-1).values.sigmoid()
        # initialize empty instance_inds
        instance_inds = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_inds is not None  # not first frame of training
            and self.instance_inds.shape[0] == instance_inds.shape[0]
        ):
            # expect both past inds and new inds to have the same shape
            assert self.instance_inds.shape[1] == instance_inds.shape[1], (
                self.instance_inds.shape,
                instance_inds.shape,
            )
            instance_inds[:, : self.instance_inds.shape[1]
                          ] = self.instance_inds
        # for instances with no ID
        mask = instance_inds < 0
        # for instances with confidence above threshold
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        # assign them new IDs
        new_ids = torch.arange(num_new_instance).to(
            instance_inds) + self.prev_id
        instance_inds[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        return instance_inds

    def update_instance_inds(self, instance_inds, confidence, topk_indices=None):
        """Prepare self.instance_inds for the next frame, appending 300 new instances of value -1 to the end (for the PQ)"""
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        # take top-k instances with highest confidence
        if topk_indices is None:
            _, instance_inds, _ = topk(
                temp_conf, self.num_temp_instances, instance_inds)
            instance_inds = instance_inds[0]
            instance_inds = instance_inds.squeeze(dim=-1)
        else:
            bs, k = topk_indices.shape
            batch_indices = torch.arange(
                bs, device=instance_inds.device).unsqueeze(-1).expand(-1, k)
            instance_inds = instance_inds[batch_indices, topk_indices]
        # pad with -1 on the end
        self.instance_inds = F.pad(
            instance_inds,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
