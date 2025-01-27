# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule

from projects.mmdet3d_plugin.models.utils.utils import (
    MLP, gen_sineembed_for_position)

__all__ = ["Sparse4DHead"]


@MODELS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        # focalformer3d params
        point_cloud_range: List[float],
        modality: str = "camera",
        multistage_heatmap: Union[int, bool] = False,
        extra_feat: bool = False,
        use_bevpos_emb: bool = True,
        xy_size: tuple = (180, 180),
        init_pq_with_heatmap: bool = False,
        nms_kernel_size: int = 3,
        # sparse4d params
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: Optional[Dict] = None,
        loss_cls: Optional[Dict] = None,
        loss_reg: Optional[Dict] = None,
        decoder: Optional[Dict] = None,
        sampler: Optional[Dict] = None,
        reg_weights: Optional[List] = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========

        self.instance_bank = MODELS.build(instance_bank)
        self.anchor_encoder = MODELS.build(anchor_encoder)
        self.sampler = MODELS.build(sampler)
        self.decoder = MODELS.build(decoder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_reg = MODELS.build(loss_reg)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model],
            "gnn": [graph_model],
            "norm": [norm_layer],
            "ffn": [ffn],
            "deformable": [deformable_model],
            "deformable_lidar": [deformable_model],
            "refine": [refine_layer],
        }
        self.layers = nn.ModuleList(
            [
                MODELS.build(self.op_config_map.get(op, None)[0])
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()
        self.num_classes = refine_layer['num_cls']
        # focalformer
        self.use_lidar = modality == "lidar"
        self.use_camera = modality == "camera"
        self.multistage_heatmap = multistage_heatmap
        if self.use_lidar:
            self.extra_feat = extra_feat
            if extra_feat:
                assert self.multistage_heatmap, "extra_feat must be used with multistage_heatmap"
            # gen_sineembed_for_position uses dim=128 for each x and y
            self.pos_embed_learned = MLP(
                128*2, self.embed_dims, self.embed_dims, 2)
            # X-min, Y-min, Z-min, X-max, Y-max, Z-max
            # used for normalizing anchor to be [0, 1] for reference_points
            self.point_cloud_range = torch.nn.Parameter(
                torch.tensor(point_cloud_range), requires_grad=False)
            self.ref_point_norm = self.point_cloud_range[3:] - \
                self.point_cloud_range[:3]
            self.use_bevpos_emb = use_bevpos_emb
            self.bev_pos = self.create_2D_grid(*xy_size)
            self.dconv = ConvModule(
                self.embed_dims, self.embed_dims,
                stride=2, kernel_size=3, padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            )
            self.dconv2 = ConvModule(
                self.embed_dims, self.embed_dims,
                stride=2, kernel_size=3, padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            )

            self.init_pq_with_heatmap = init_pq_with_heatmap
            if self.init_pq_with_heatmap:
                assert self.multistage_heatmap, "init_pq_with_heatmap must be used with multistage_heatmap"
            if init_pq_with_heatmap:
                self.create_heatmap_head()
            self.nms_kernel_size = nms_kernel_size

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def create_2D_grid(self, x_size: int, y_size: int):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def create_heatmap_head(self,):
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
        self.heatmap_head = nn.Sequential(*layers)

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        pts_inputs,
        feature_maps: Union[torch.Tensor, List],
        timestamp: torch.Tensor,
        projection_mat: torch.Tensor,
        batch_data_samples: List[Det3DDataSample],
        image_wh: Optional[torch.Tensor] = None,
    ):
        batch_metas = [item.metainfo for item in batch_data_samples]
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        if self.use_camera:
            batch_size = feature_maps[0].shape[0]
        else:
            assert self.use_lidar
            batch_size = pts_inputs[0].shape[0]
            # preprocess LiDAR features, following FocalFormer3D focal_decoder.py
            lidar_feat = pts_inputs[0]
            if self.extra_feat:
                extra_feats = pts_inputs[1][-1]
                pts_inputs[1].pop(-1)
            lidar_feat_flatten = lidar_feat.view(
                batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
            bev_pos = self.bev_pos.repeat(
                batch_size, 1, 1).to(lidar_feat.device)
            bev_pos_2 = self.create_2D_grid(
                lidar_feat.shape[2] // 2, lidar_feat.shape[2] // 2).repeat(batch_size, 1, 1).to(lidar_feat.device) * 2
            bev_pos_4 = self.create_2D_grid(
                lidar_feat.shape[2] // 4, lidar_feat.shape[2] // 4).repeat(batch_size, 1, 1).to(lidar_feat.device) * 4

        # TODO move the following code to a separate function
        if not self.multistage_heatmap and self.use_lidar:
            if self.init_pq_with_heatmap:
                dense_heatmap = self.heatmap_head(lidar_feat)
            # iterbev_wo_img always true in head in FocalFormer3D
            if isinstance(pts_inputs[1], (list, tuple)):
                # if using extra_feat
                new_lidar_feat = pts_inputs[1][-1]
            else:  # no extra_feat
                new_lidar_feat = pts_inputs[1]
            lidar_feat_flatten = new_lidar_feat.view(
                *lidar_feat_flatten.shape)
            if self.init_pq_with_heatmap:
                # do heatmap PQ initialization
                dense_heatmap_img = self.heatmap_head_img(
                    new_lidar_feat.view(lidar_feat.shape))  # [BS, num_classes, H, W]
                # average both heatmaps
                heatmap = (dense_heatmap.detach().sigmoid() +
                           dense_heatmap_img.detach().sigmoid()) / 2

                if self.use_camera or self.iterbev_wo_img:
                    heatmap_train = [dense_heatmap, dense_heatmap_img]
                else:
                    heatmap_train = dense_heatmap

                padding = self.nms_kernel_size // 2
                local_max = torch.zeros_like(heatmap)
                # equals to nms radius = voxel_size * out_size_factor * kenel_size
                local_max_inner = F.max_pool2d(
                    heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
                local_max[:, :, padding:(-padding),
                          padding:(-padding)] = local_max_inner
                # for Pedestrian & Traffic_cone in nuScenes
                if self.test_cfg['dataset'] == 'nuScenes':
                    local_max[:, 8, ] = F.max_pool2d(
                        heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                    local_max[:, 9, ] = F.max_pool2d(
                        heatmap[:, 9], kernel_size=1, stride=1, padding=0)
                # for Pedestrian & Cyclist in Waymo
                elif self.test_cfg['dataset'] == 'Waymo':
                    local_max[:, 1, ] = F.max_pool2d(
                        heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                    local_max[:, 2, ] = F.max_pool2d(
                        heatmap[:, 2], kernel_size=1, stride=1, padding=0)
                heatmap = heatmap * (heatmap == local_max)
                heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

                # top #num_proposals among all classes
                top_proposals = heatmap.view(
                    batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
                top_proposals_class = top_proposals // heatmap.shape[-1]
                top_proposals_index = top_proposals % heatmap.shape[-1]
                query_feat = lidar_feat_flatten.gather(
                    index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
                self.query_labels = top_proposals_class

                # add category embedding
                one_hot = F.one_hot(top_proposals_class,
                                    num_classes=self.num_classes).permute(0, 2, 1)
                query_cat_encoding = self.class_encoding(one_hot.float())
                query_feat += query_cat_encoding

                query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(
                    0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
                query_heatmap_score = heatmap.gather(
                    index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)
        elif self.use_lidar:  # multistage_heatmap, capture hard FN
            dense_heatmap = self.heatmap_head(lidar_feat)  # original

            multistage_feats = pts_inputs[1]
            if self.reuse_first_heatmap:
                multistage_feats.insert(0, lidar_feat)

            query_labels = []
            query_feats = []
            query_poses = []
            query_heatmap_scores = []
            acc_masks = torch.ones_like(dense_heatmap).view(batch_size, -1)
            multistage_masks = []
            multistage_masks_independent_visualize = []
            heatmap_train = []
            for i in range(self.multistage_heatmap):
                if i == 0 and self.reuse_first_heatmap:
                    # do not support heatmap_box, see FocalFormer3D

                    heatmap = dense_heatmap.detach().sigmoid()
                    heatmap_train.append(dense_heatmap)
                    multistage_masks.append(
                        acc_masks.view(*heatmap.shape).clone())
                    # remove early positive
                    heatmap = heatmap * acc_masks.view(*heatmap.shape)
                else:
                    dense_heatmap_img = self.heatmap_head_img[i](
                        multistage_feats[i])
                    # do not support heatmap_box, see FocalFormer3D

                    heatmap = dense_heatmap_img.detach().sigmoid()
                    if i == 0:
                        heatmap_train.append(dense_heatmap)
                        multistage_masks.append(
                            acc_masks.view(*heatmap.shape).clone())
                    # remove early positive
                    heatmap = heatmap * acc_masks.view(*heatmap.shape)
                    heatmap_train.append(dense_heatmap_img)
                    multistage_masks.append(
                        acc_masks.view(*heatmap.shape).clone())

                lidar_feat_flatten = multistage_feats[i].view(
                    *lidar_feat_flatten.shape)

                padding = self.nms_kernel_size // 2
                local_max = torch.zeros_like(heatmap)
                # equals to nms radius = voxel_size * out_size_factor * kenel_size
                local_max_inner = F.max_pool2d(
                    heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
                local_max[:, :, padding:(-padding),
                          padding:(-padding)] = local_max_inner
                # for Pedestrian & Traffic_cone in nuScenes
                if self.test_cfg['dataset'] == 'nuScenes':
                    local_max[:, 8, ] = F.max_pool2d(
                        heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                    local_max[:, 9, ] = F.max_pool2d(
                        heatmap[:, 9], kernel_size=1, stride=1, padding=0)
                # for Pedestrian & Cyclist in Waymo
                elif self.test_cfg['dataset'] == 'Waymo':
                    local_max[:, 1, ] = F.max_pool2d(
                        heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                    local_max[:, 2, ] = F.max_pool2d(
                        heatmap[:, 2], kernel_size=1, stride=1, padding=0)
                heatmap = heatmap * (heatmap == local_max)
                heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

                # top #num_proposals among all classes
                top_proposals = torch.topk(heatmap.view(
                    batch_size, -1), k=self.num_proposals, dim=-1, largest=True, sorted=False).indices
                # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
                top_proposals_class = top_proposals // heatmap.shape[-1]
                top_proposals_index = top_proposals % heatmap.shape[-1]
                query_feat = lidar_feat_flatten.gather(
                    index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)

                query_labels.append(top_proposals_class)

                # add category embedding
                one_hot = F.one_hot(
                    top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
                query_cat_encoding = self.class_encoding(one_hot.float())

                query_feat += query_cat_encoding
                query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(
                    0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
                query_heatmap_score = heatmap.gather(
                    index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)

                query_feats.append(query_feat)
                query_poses.append(query_pos)
                query_heatmap_scores.append(query_heatmap_score)

                # do not support heatmap_box, see FocalFormer3D

                ################ select to ignore ######################
                # only use mask_heatmap_mode=='poscls', following FocalFormer3D
                selected_mask = acc_masks.new_zeros(
                    batch_size, self.num_classes * heatmap.shape[-1])
                selected_mask.scatter_(index=top_proposals, dim=1, src=torch.ones_like(
                    top_proposals, dtype=acc_masks.dtype))

                selected_mask = selected_mask.reshape(*dense_heatmap.shape)
                # masking by pooling
                selected_mask_kernel = F.max_pool2d(
                    selected_mask, kernel_size=self.nms_kernel_size, stride=1, padding=self.nms_kernel_size // 2)
                # for Pedestrian & Traffic_cone in nuScenes
                if self.test_cfg['dataset'] == 'nuScenes':
                    selected_mask_kernel[:, 8:10] = F.max_pool2d(
                        selected_mask[:, 8:10], kernel_size=1, stride=1, padding=0)
                # for Pedestrian & Cyclist in Waymo
                elif self.test_cfg['dataset'] == 'Waymo':
                    selected_mask_kernel[:, 1:3] = F.max_pool2d(
                        selected_mask[:, 1:3], kernel_size=1, stride=1, padding=0)

                acc_masks = acc_masks * \
                    (1.-selected_mask_kernel).view(*acc_masks.shape)

            self.query_labels = torch.cat(query_labels, dim=1)
            query_feat = torch.cat(query_feats, dim=2)
            query_pos = torch.cat(query_poses, dim=1)
            query_heatmap_score = torch.cat(query_heatmap_scores, dim=2)

            self.num_proposals = self.num_proposals_ori * self.multistage_heatmap

        if self.use_lidar:
            if self.init_pq_with_heatmap:
                query_labels = self.query_labels

            # skip focal former DN generation
            # Always do multiscale
            if not self.multistage_heatmap:
                lidar_feat = new_lidar_feat
            else:
                if self.extra_feat:
                    lidar_feat = extra_feats
                else:
                    lidar_feat = multistage_feats[-1]

            multiscale_inputs = [lidar_feat]
            multiscale_inputs.append(self.dconv(multiscale_inputs[-1]))
            multiscale_inputs.append(
                self.dconv2(multiscale_inputs[-1]))
            multiscale_inputs_flatten = torch.cat(
                [i.flatten(2, 3) for i in multiscale_inputs], dim=-1)
            # moved the line below out of the decoder layer for loop in focal_decoder
            bev_pos = torch.cat([bev_pos, bev_pos_2, bev_pos_4], dim=1)

            ################## Deformable Parameters #############
            # always do multiscale
            spatial_shapes = torch.as_tensor(
                [i.shape[2:] for i in multiscale_inputs], dtype=torch.long, device='cuda')
            level_start_index = torch.as_tensor(
                [0, *(torch.cumsum(torch.prod(spatial_shapes, dim=1), dim=0)[:-1])], dtype=torch.long, device='cuda')

            # lidar feat
            lidar_feat_flatten = multiscale_inputs_flatten
            MSDA_kwargs = dict(
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=torch.ones((batch_size, 1, 2), device='cuda'),
            )

            if self.use_bevpos_emb:
                bev_reference_points = bev_pos / \
                    torch.flip(spatial_shapes[:1], dims=(1,))[:, None]
                bev_sine_pos = gen_sineembed_for_position(
                    bev_reference_points[:, :, :2])  # B, total num bev_poses, self.embed_dims
                bev_pos_embed = self.pos_embed_learned(
                    bev_sine_pos)  # bs, nq, 256
                # TODO( multiple addition for bev pos embedding )
                pos_lidar_feat_flatten = lidar_feat_flatten + \
                    bev_pos_embed.transpose(1, 2)
            else:
                pos_lidar_feat_flatten = lidar_feat_flatten

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        # more precise inversion
        batched_global2lidar = []
        for x in batch_metas:
            g2l = x["lidar2global"].new_zeros((4, 4))
            g2l[:3, :3] = x["lidar2global"][:3, :3].T
            g2l[:3, 3] = -g2l[:3, :3] @ x["lidar2global"][:3, 3]
            g2l[3, 3] = 1
            batched_global2lidar.append(g2l)
        (
            instance_feature,
            anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size,
            timestamp,
            batched_global2lidar=batched_global2lidar,
            dn_metas=self.sampler.dn_metas
        )

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            dn_metas = self.sampler.get_dn_anchors(
                [ds.gt_instances_3d.labels_3d for ds in batch_data_samples],
                [ds.gt_instances_3d.bboxes_3d for ds in batch_data_samples],
                [ds.gt_instances_3d.instance_inds for ds in batch_data_samples],
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                # attend to learnable instances (300) + temp instances (600) or in first frame case only learnable instances (900)
                # in temp_gnn, do not attend to dn instances
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature[:, :self.instance_bank.num_anchor],
                    instance_feature[:, :self.instance_bank.num_anchor],
                    query_pos=anchor_embed,
                    key_pos=anchor_embed[:, :self.instance_bank.num_anchor],
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    projection_mat,
                    image_wh,
                )
            elif op == "deformable_lidar":
                # normalize anchor to [0, 1] to get reference_points
                reference_points = anchor[..., :2]
                reference_points = (reference_points - self.point_cloud_range[:2].to(
                    reference_points.device)) / self.ref_point_norm[:2].to(reference_points.device)
                reference_points = reference_points.clamp(0, 1)
                # expand reference_points to match the shape of valid ratios, see mmdet DeformableDetrTransformerDecoder
                reference_points_input = \
                    reference_points[:, :, None] * \
                    MSDA_kwargs["valid_ratios"][:, None]
                output = self.layers[i](
                    query=instance_feature,  # B x N x C
                    value=pos_lidar_feat_flatten.permute(
                        0, 2, 1),  # B C Pv -> B Pv C
                    key=pos_lidar_feat_flatten.permute(  # only used by dense attn
                        0, 2, 1),  # B C Pv -> B Pv C
                    query_pos=anchor_embed,  # B N C
                    reference_points=reference_points_input,  # B N num_levels 2
                    **MSDA_kwargs,)
                # follow DeformableFeatureAggregation, concat the output
                instance_feature = torch.cat(
                    [instance_feature, output], dim=-1)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    # update anchor_embed for the next transformer block based on output of the refinement layer
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        if not self.training:
            # assign instance_inds to all predictions for inference
            instance_inds = self.instance_bank.get_instance_ind(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_inds"] = instance_inds
        else:
            output["instance_inds"] = None
        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature,
            anchor,
            cls,
            timestamp,
            [x["lidar2global"] for x in batch_metas],
            output["instance_inds"]
        )
        return output

    def loss(self, model_outs, batch_data_samples):
        gt_cls = [bs.gt_instances_3d.labels_3d for bs in batch_data_samples]
        gt_reg = [bs.gt_instances_3d.bboxes_3d for bs in batch_data_samples]
        gt_id = [bs.gt_instances_3d.instance_inds for bs in batch_data_samples]
        num_gt = [len(x) for x in gt_cls]
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            # TODO move code in this for loop to a separate function
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights, id_target = self.sampler.sample(
                cls,
                reg,
                gt_cls,
                gt_reg,
                gt_id
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls_flattened = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(
                cls_flattened, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

            # compute metrics for query consistency
            qc_metrics = []
            prev_instance_inds = self.instance_bank.instance_inds_training
            if prev_instance_inds is None:
                prev_instance_inds = [None for i in range(cls.shape[0])]
            else:
                # if not mask, set to None
                prev_instance_inds = [
                    prev_instance_inds[bs] if self.instance_bank.mask[bs] else None
                    for bs in range(cls.shape[0])
                ]
            confidences = cls.max(dim=-1).values.sigmoid()
            for bs, (gt_id_i, conf_i, id_target_i, prev_instance_inds_i) in enumerate(zip(gt_id, confidences, id_target, prev_instance_inds)):
                qc_metrics.append(self.compute_qc_metrics(
                    gt_id_i, conf_i, id_target_i, prev_instance_inds_i))
            for key in qc_metrics[0].keys():
                val = [x[key] for x in qc_metrics]
                val = torch.stack(val).nanmean()  # account for nan entries
                if not val.isnan():
                    # add decoder suffix to qc metrics
                    output["qc_metrics/"+key+f"_{decoder_idx}"] = val

        # for the final layer, cache the id_target for the next timestep
        # assuming non-zero decoder layers
        bs, k = self.instance_bank.cached_indices.shape
        batch_indices = torch.arange(
            bs, device=id_target.device).unsqueeze(-1).expand(-1, k)
        # cache id target to intsance_inds for the next timestep
        self.instance_bank.instance_inds_training = id_target[batch_indices, self.instance_bank.cached_indices]
        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def compute_qc_metrics(self, gt_id, conf, id_target, prev_instance_inds=None):
        if prev_instance_inds is not None:
            num_temp_instances = self.instance_bank.num_temp_instances
        else:
            num_temp_instances = 0
        tq_conf, pq_conf = conf[:num_temp_instances], conf[num_temp_instances:]
        tq_id_target, pq_id_target = id_target[:
                                               num_temp_instances], id_target[num_temp_instances:]
        # num_gt = gt_id.shape[0] # for debugging purposes

        if prev_instance_inds is not None:
            # Convert instance_inds to a tensor and filter out -1 values
            valid_prev_instance_inds = prev_instance_inds[prev_instance_inds != -1]
            # how does this work with different batch sizes, num gt?
        else:
            valid_prev_instance_inds = torch.empty(
                (conf.shape[0], 0), dtype=torch.long, device=conf.device)

        # Create a mask for which pq were in prev frame
        prev_pq_mask = torch.zeros_like(
            pq_id_target, dtype=torch.bool)  # (num_pq)
        prev_pq_mask = torch.isin(pq_id_target, valid_prev_instance_inds)

        # Create a mask for current IDs, check not -1
        pos_pq_mask = pq_id_target != -1

        # Newborn mask is true where the pq pred is pos in curr but not in prev_mask
        newborn_mask = pos_pq_mask & ~prev_pq_mask
        # pq_tp is true when the pq is assigned (!=-1) and it is a newborn gt (not in prev frame)
        pq_tp = newborn_mask.sum()
        # pq_fp: pq assigned but it is not a newborn gt (it was a prev tracked obj)
        pq_fp = pos_pq_mask.sum() - pq_tp
        # pq_fn: a newborn gt that was assigned to a tq, not a hinderance to query consistency, ignore
        metric_dict = dict(
            pq_tp_conf=pq_conf[newborn_mask].mean(),
            pq_fp_conf=pq_conf[pos_pq_mask & prev_pq_mask].mean(),
            pq_neg_conf=pq_conf[~pos_pq_mask].mean(),
            # of the total pos pq predictions, how many were actual newborn obj
            pq_precision=pq_tp/(pq_tp+pq_fp),
        )
        if num_temp_instances > 0:
            pos_tq_mask = tq_id_target != -1
            # tq_tp: tq assigned (!=-1) and it was the same previously tracked obj
            tq_tp_mask = (tq_id_target == prev_instance_inds) & pos_tq_mask
            tq_tp = tq_tp_mask.sum()
            # tq_fp: tq assigned (!=-1) and it was a newborn gt OR it was a different gt
            tq_fp_mask = pos_tq_mask & (tq_id_target != prev_instance_inds)
            tq_fp = tq_fp_mask.sum()
            # tq_fn: the gt is in current frame but not assigned to the same TQ
            tq_fn_mask = torch.isin(prev_instance_inds, gt_id) & ~tq_tp_mask
            tq_fn = tq_fn_mask.sum()
            metric_dict.update(
                tq_tp_conf=tq_conf[tq_tp_mask].mean(),
                tq_fp_conf=tq_conf[tq_fp_mask].mean(),
                tq_fn_conf=tq_conf[tq_fn_mask].mean(),
                # of the total pos tq predictions, how many were actual prev tracked obj
                tq_precision=tq_tp/(tq_tp+tq_fp),
                # of the total tracked obj that are also in current frame, how many maintained query consistency
                tq_recall=tq_tp/(tq_tp+tq_fn),
            )
        else:
            device = pq_conf.device
            metric_dict.update(
                tq_tp_conf=torch.tensor(torch.nan, device=device),
                tq_fp_conf=torch.tensor(torch.nan, device=device),
                tq_fn_conf=torch.tensor(torch.nan, device=device),
                tq_precision=torch.tensor(torch.nan, device=device),
                tq_recall=torch.tensor(torch.nan, device=device),
            )
        return metric_dict

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_inds"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
