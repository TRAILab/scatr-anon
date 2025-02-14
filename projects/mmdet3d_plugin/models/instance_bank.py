import copy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Linear, Scale, build_conv_layer
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
from torch import nn

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import \
    SparseBox3DDecoder
from projects.mmdet3d_plugin.models.utils.utils import get_dense_grid_points

from .blocks import linear_relu_ln
from .constants import UNTRACKED_ID

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
        # (bs, num_groups, k, ...)
        selected_elements = input_i[batch_indices, group_indices, indices]
        outputs.append(selected_elements)
    return confidence, outputs, indices  # Return indices as well


@MODELS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor: int,
        embed_dims: int,
        anchor: str,
        class_names: List[str],
        anchor_handler=None,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        anchor_grad: bool = True,
        feat_grad: bool = True,
        max_time_interval: float = 2,
        # TRAILAB params
        num_learned_groups: int = 1,
        num_learned_temp_groups: int = 1,
        # FocalFormer3D heatmap init params
        heatmap_init: bool = False,
        num_heatmap_stages: int = 1,
        xy_size: tuple = (180, 180),
        point_cloud_range: List[float] = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        nms_kernel_size: int = 3,
        num_bbox_pool_points: int = 7,
        dataset_name:str='NuScenesTrackingDataset',
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

        self.num_anchor = num_anchor

        """
        If not heatmap_init, anchor is (900, 4) representing 900 xy, wl boxes in BEV
        """
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        else:
            assert isinstance(anchor, np.ndarray), (
                f"Anchor must be a path to a .npy file, a list, or a numpy array, not {type(anchor)}"
            )
        if not heatmap_init:
            assert anchor.shape[0] == num_anchor, (
                f"Anchor shape {anchor.shape} does not match num_anchor {num_anchor}"
            )
            assert anchor.shape[1] == 11, "Anchor shape must be (num_anchor, 11)"

        self.anchor = nn.Parameter(
            torch.from_numpy(anchor).float(),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor

        if not heatmap_init:
            self.instance_feature = nn.Parameter(
                torch.zeros(
                    [num_learned_groups, self.anchor.shape[0], self.embed_dims]),
                requires_grad=feat_grad,
            )
        else:
            self.instance_feature = None
        self.num_learned_groups = num_learned_groups
        self.num_learned_temp_groups = num_learned_temp_groups
        self.reset()

        # FocalFormer3D heatmap init params
        self.heatmap_init = heatmap_init
        if heatmap_init and num_learned_groups > 1:
            raise NotImplementedError(
                "Only single group is supported for heatmap init")
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.num_heatmap_stages = num_heatmap_stages
        if self.num_heatmap_stages > 0:
            self.num_heatmap_stages += 1  # account for reusing original heatmap
        self.point_cloud_range = torch.tensor(point_cloud_range)
        # convert to parameter for automatic device handling
        self.point_cloud_range = nn.Parameter(
            self.point_cloud_range, requires_grad=False)
        self.xy_size = xy_size
        self.nms_kernel_size = nms_kernel_size
        self.nms_padding = nms_kernel_size // 2
        self.num_bbox_pool_points = num_bbox_pool_points
        self.dataset_name = dataset_name
        if self.heatmap_init:
            assert self.num_heatmap_stages > 0
            self.create_heatmap_head()
            assert 'nuscenes' in self.dataset_name.lower() or 'waymo' in self.dataset_name.lower(), \
                f"Dataset {self.dataset_name} not supported for heatmap init"

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if not self.heatmap_init and self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def create_heatmap_head(self):
        """
        Top level list: groups (generally 5 groups)
        Second level list: number of multi-stage heatmaps (1 + 1 for lidar, 1+2 for fusion)
        """
        # self.heatmap_head = nn.ModuleList()
        # for _ in range(self.num_learned_groups):
        layers = []
        layers.append(ConvModule(
            self.embed_dims,
            self.embed_dims,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        ))
        layers.append(build_conv_layer(
            dict(type='Conv2d'),
            self.embed_dims,
            self.num_classes,
            kernel_size=3,
            padding=1,
            bias='auto',
        ))
        self.heatmap_head = nn.ModuleList([
            copy.deepcopy(nn.Sequential(*layers))
            for _ in range(self.num_heatmap_stages)])
        self.cat_encoding = nn.Conv1d(self.num_classes, self.embed_dims, 1)
        self.heatmap_bbox_head = nn.Sequential(
            *linear_relu_ln(self.embed_dims, 2, 2),
            Linear(self.embed_dims, 11),
            Scale([1.0] * 11),  # bbox_dim = 11
        )

        fc_list = []
        pre_channel = self.num_bbox_pool_points ** 2 * \
            self.embed_dims * (3)  # 3 levels of multiscale inputs
        num_roi_layers = 3
        for i in range(num_roi_layers):
            chl = self.embed_dims * 4 if i < num_roi_layers - 1 else self.embed_dims
            fc_list.extend([
                nn.Linear(pre_channel, chl, bias=False),
                nn.BatchNorm1d(chl),
                nn.ReLU(inplace=True)
            ])
            fc_list.append(nn.Dropout(0.1))
            pre_channel = chl
        self.roi_mlp = nn.Sequential(*fc_list)

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

    def get_pq_learned(self, batch_size):
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
        return instance_feature, anchor

    def get_pq_heatmap(self, batch_size: int, multistage_feats, multiscale_lidar_feats, bev_pos):
        # also check sizes of multistage_feats
        assert len(multistage_feats) == self.num_heatmap_stages  # sanity check
        num_groups = self.num_learned_groups if self.training else 1
        assert self.num_anchor % self.num_heatmap_stages == 0, "num_anchor must be divisible by num_stages"
        num_proposals_per_stage = self.num_anchor // self.num_heatmap_stages
        feat_w, feat_h = self.xy_size
        query_feats_list = []
        query_poses_list = []
        query_cat_encoding_list = []
        acc_masks = torch.ones(
            (batch_size, self.num_classes * feat_h * feat_w), 
            device=multistage_feats[0].device)
        multistage_acc_masks = []
        dense_heatmap_list = []
        # iterate through each heatmap
        for head, feats in zip(self.heatmap_head, multistage_feats):
            # pass dense_heatmap through each heatmap in group
            dense_heatmap = head(feats)
            dense_heatmap_list.append(dense_heatmap)
            multistage_acc_masks.append(acc_masks.view(*dense_heatmap.shape).clone())

            # remove early positive in heatmap using acc_masks
            heatmap = dense_heatmap.detach().sigmoid()
            heatmap = heatmap * acc_masks.view(*heatmap.shape)

            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kernel_size
            local_max_inner = F.max_pool2d(
                heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, self.nms_padding:(-self.nms_padding),
                      self.nms_padding:(-self.nms_padding)] = local_max_inner
            # for Pedestrian & Traffic_cone in nuScenes
            if 'nuscenes' in self.dataset_name.lower():
                ped_idx = self.class_names.index('pedestrian')
                traffic_cone_idx = self.class_names.index('traffic_cone')
                local_max[:, ped_idx, ] = F.max_pool2d(
                    heatmap[:, ped_idx], kernel_size=1, stride=1, padding=0)
                local_max[:, traffic_cone_idx, ] = F.max_pool2d(
                    heatmap[:, traffic_cone_idx], kernel_size=1, stride=1, padding=0)
            # for Pedestrian & Cyclist in Waymo
            elif 'waymo' in self.dataset_name.lower():
                ped_idx = self.class_names.index('Pedestrian')
                cyclist_idx = self.class_names.index('Cyclist')
                local_max[:, ped_idx, ] = F.max_pool2d(
                    heatmap[:, ped_idx], kernel_size=1, stride=1, padding=0)
                local_max[:, cyclist_idx, ] = F.max_pool2d(
                    heatmap[:, cyclist_idx], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

            # top #num_proposals among all classes
            top_proposals = torch.topk(
                heatmap.view(batch_size, -1),
                k=num_proposals_per_stage,
                dim=-1,
                largest=True,
                sorted=False
            ).indices

            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            # (bs, embed_dims, feat_h * feat_w)
            lidar_feat_flatten = feats.reshape(
                batch_size, self.embed_dims, -1)
            query_feat = lidar_feat_flatten.gather(
                index=top_proposals_index[:, None,
                                          :].expand(-1, lidar_feat_flatten.shape[1], -1),
                dim=-1)

            # add category embedding
            one_hot = F.one_hot(
                top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.cat_encoding(one_hot.float())
            query_feat += query_cat_encoding

            query_pos = bev_pos.gather(
                index=top_proposals_index[:, None, :].permute(
                    0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
                dim=1)

            query_feats_list.append(query_feat)
            query_poses_list.append(query_pos)
            query_cat_encoding_list.append(query_cat_encoding)

            ################ select to ignore for next stage ######################
            selected_mask = acc_masks.new_zeros(
                batch_size,
                self.num_classes * heatmap.shape[-1])
            selected_mask.scatter_(
                index=top_proposals,
                dim=1,
                src=torch.ones_like(
                    top_proposals,
                    dtype=acc_masks.dtype
                )
            )
            selected_mask = selected_mask.reshape(*dense_heatmap.shape)

            # masking by pooling
            selected_mask_kernel = F.max_pool2d(
                selected_mask,
                kernel_size=self.nms_kernel_size,
                stride=1,
                padding=self.nms_padding)
            # for Pedestrian & Traffic_cone in nuScenes
            if 'nuscenes' in self.dataset_name.lower():
                selected_mask_kernel[:, ped_idx] = F.max_pool2d(
                    selected_mask[:, ped_idx], kernel_size=1, stride=1, padding=0)
                selected_mask_kernel[:, traffic_cone_idx] = F.max_pool2d(
                    selected_mask[:, traffic_cone_idx], kernel_size=1, stride=1, padding=0)
            # for Pedestrian & Cyclist in Waymo
            elif 'waymo' in self.dataset_name.lower():
                selected_mask_kernel[:, ped_idx] = F.max_pool2d(
                    selected_mask[:, ped_idx], kernel_size=1, stride=1, padding=0)
                selected_mask_kernel[:, cyclist_idx] = F.max_pool2d(
                    selected_mask[:, cyclist_idx], kernel_size=1, stride=1, padding=0)

            acc_masks = acc_masks * \
                (1.0 - selected_mask_kernel).view(*acc_masks.shape)

        # query feat has num proposals at the end due to the use of conv layers instead of linear
        # bs, embed_dims, num_proposals
        query_feat = torch.cat(query_feats_list, dim=2).transpose(1, 2)
        query_pos = torch.cat(query_poses_list, dim=1)  # bs, num_proposals, 2
        query_cat_encoding = torch.cat(
            query_cat_encoding_list, dim=2).transpose(1, 2)  # bs, num_proposals, embed_dims
        # check query_pos: is it in (0, 180) or is it in real world coordinates?

        # convert query_pos from (0, 180) back to point_cloud_range
        query_pos = query_pos / torch.tensor(
            [self.xy_size[1], self.xy_size[0]], device=query_pos.device)
        query_pos = query_pos * (self.point_cloud_range[3] - self.point_cloud_range[0]) + \
            self.point_cloud_range[[0, 1]].view(1, 2)

        # bs, num_proposals, embed_dims
        anchors = self.heatmap_bbox_head(query_feat)  # bs, num_proposals, 10
        anchors[:, :, [X, Y]] += query_pos  # add xy offset

        # use predicted anchors to sample lidar feats for instance_feats
        instance_feats = InstanceBank.bbox_feat_pooling(
            anchors,
            multiscale_lidar_feats,
            self.roi_mlp,
            self.point_cloud_range,
            self.num_bbox_pool_points,
            self.embed_dims,
        )
        instance_feats += query_cat_encoding  # add category encoding

        # expand an extra dimension for groups, TODO support multiple groups
        instance_feats = instance_feats.unsqueeze(1)
        anchors = anchors.unsqueeze(1)

        return instance_feats, anchors, dense_heatmap_list, multistage_acc_masks

    @staticmethod
    def bbox_feat_pooling(
            anchors: torch.Tensor,
            multiscale_lidar_feats,
            roi_mlp,
            point_cloud_range,
            roi_feats,
            embed_dims,
    ):
        """
        anchors: (bs, num_groups, num_proposals, 11)
        lidar_feats: (bs, num_groups, num_proposals, embed_dims)
        """
        batch_size, num_proposals = anchors.shape[:2]
        # decode anchors into proper space
        std_boxes = SparseBox3DDecoder.decode_box(anchors)  # (bs, num_proposals, 7)
        std_boxes = std_boxes.reshape(batch_size * num_proposals, std_boxes.shape[-1]) # (bs * num_proposals, 7)
        # use decoded anchors to generate dense grid points
        grid_points = get_dense_grid_points(
            std_boxes,
            roi_feats)
        # add z dim
        grid_points = torch.cat(
            [grid_points, grid_points.new_ones(*grid_points.shape[:2], 1)], dim=-1)
        # rotate grid points
        grid_points = rotation_3d_in_axis(
            grid_points, std_boxes[:, SIN_YAW], axis=2)  # (bs * num_proposals, roi_feats**2, 3)
        # add grid points to decoded anchors
        grid_points = grid_points[..., [X, Y]] + std_boxes[:, None, [X, Y]]

        # reshape grid_points for the normalization
        grid_points = grid_points.view(
            batch_size,
            num_proposals,
            roi_feats**2,
            2
        )
        # normalize from world space (-54, 54) to grid sample dims (-1, 1)
        grid_points = (grid_points - point_cloud_range[:2]) / (
            point_cloud_range[3:5] - point_cloud_range[:2])
        grid_points = grid_points * 2. - 1.
        grid_points = grid_points.clip(min=-2., max=2.)

        # extract features at grid points
        ms_roi_feat = []
        for feat in multiscale_lidar_feats:
            # roi_feat (bs, embed_dim, num_proposals, roi_feats**2)
            roi_feat = F.grid_sample(feat, grid_points)
            # roi_feat = torch.zeros(batch_size, embed_dims, num_proposals, roi_feats**2).cuda()
            ms_roi_feat.append(roi_feat)
        # (bs, embed_dim * num multiscale_inputs, num_proposals, roi_feats**2)
        roi_feat = torch.cat(ms_roi_feat, dim=1)

        # reshape for roi_mlp
        # (bs, embed_dim * num_scales, num_proposals, roi_feats**2)
        # -> (bs, num_proposals, embed_dim * num_scales, roi_feats**2)
        # -> (bs * num_proposals, embed_dim * num_scales * roi_feats**2)
        roi_feat = roi_feat.permute(0, 2, 1, 3).reshape(
            batch_size * num_proposals,
            embed_dims * len(multiscale_lidar_feats) * roi_feats**2)
        # pass extracted feats through roi_mlp
        roi_feat = roi_mlp(roi_feat)
        # reshape back
        roi_feat = roi_feat.view(batch_size, num_proposals, embed_dims)
        return roi_feat

    def get(self, batch_size, timestamp, batched_global2lidar, dn_metas=None, multistage_feats=None, multiscale_lidar_feats=None, bev_pos=None):
        if self.heatmap_init:
            instance_feature, anchor, dense_heatmap_list, acc_masks = self.get_pq_heatmap(
                batch_size, multistage_feats, multiscale_lidar_feats, bev_pos)
        else:
            instance_feature, anchor = self.get_pq_learned(batch_size)
            dense_heatmap_list, acc_masks = None, None

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
            # instance_feature groups matches cached groups
            and instance_feature.shape[1] == self.cached_anchor.shape[1]
        ):
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
                    # flatten to (bs, num_groups * num_anchor, box size)
                    self.cached_anchor.flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0].reshape(self.cached_anchor.shape)

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
            dense_heatmap_list,
            acc_masks,
        )

    def update(self, instance_feature, anchor, cls):
        if self.cached_feature is None or self.cached_feature.shape[1] != instance_feature.shape[1]:
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
                self.cached_confidence * self.confidence_decay,  # cached confidence * decay
                # current confidence
                confidence[:, :, : self.num_temp_instances],
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

    def get_instance_ind(self, cls_pred: torch.Tensor, threshold: float = 0.0):
        # convert class prediction to confidence
        confidence = cls_pred.max(dim=-1).values.sigmoid()
        # initialize empty instance_inds
        instance_inds = confidence.new_full(
            confidence.shape, UNTRACKED_ID, dtype=torch.long)
        if (
            self.instance_inds_inference is not None  # not first frame of sequence
            and self.instance_inds_inference.shape[0] == instance_inds.shape[0]
        ):
            # expect both past inds and new inds to have the same shape
            assert self.instance_inds_inference.shape[2] == instance_inds.shape[2], (
                self.instance_inds_inference.shape,
                instance_inds.shape,
            )  # sanity check
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
        instance_inds = instance_inds[batch_indices,
                                      group_indices, topk_indices]
        # pad with -1 on the end
        self.instance_inds_inference = F.pad(
            instance_inds,
            (0, self.num_anchor - self.num_temp_instances),
            value=UNTRACKED_ID,
        )
