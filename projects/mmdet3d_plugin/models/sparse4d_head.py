# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule

from projects.mmdet3d_plugin.models.utils.utils import (
    MLP, gen_sineembed_for_position)

from .constants import NEG_DN_CLS_TARGET, PAD_CLS_TARGET, UNTRACKED_ID

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
        use_bevpos_emb: bool = True,
        xy_size: tuple = (180, 180),
        # sparse4d params
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: Optional[Dict] = None,
        loss_cls: Optional[Dict] = None,
        loss_reg: Optional[Dict] = None,
        loss_heatmap: Optional[Dict] = None,
        loss_heatmap_reg: Optional[Dict] = None,
        decoder: Optional[Dict] = None,
        sampler: Optional[Dict] = None,
        reg_weights: Optional[List[float]] = None,
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
        if not isinstance(self.reg_weights, torch.Tensor):
            self.reg_weights = torch.tensor(self.reg_weights)

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
            self.operation_order = operation_order[3:]
        else:
            self.operation_order = operation_order

        # =========== build modules ===========

        instance_bank['point_cloud_range'] = point_cloud_range
        self.instance_bank = MODELS.build(instance_bank)
        sampler['point_cloud_range'] = point_cloud_range
        sampler['embed_dims'] = self.instance_bank.embed_dims
        self.anchor_encoder = MODELS.build(anchor_encoder)
        self.sampler = MODELS.build(sampler)
        self.decoder = MODELS.build(decoder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_reg = MODELS.build(loss_reg)
        self.loss_heatmap = MODELS.build(loss_heatmap)
        self.loss_heatmap_reg = MODELS.build(loss_heatmap_reg)
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
        if self.use_lidar:
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
            self.bev_pos_1 = self.create_2D_grid(*xy_size)
            self.bev_pos_2 = self.create_2D_grid(
                xy_size[0] // 2, xy_size[1] // 2) * 2
            self.bev_pos_4 = self.create_2D_grid(
                xy_size[0] // 4, xy_size[1] // 4) * 4
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
            multistage_feats = None
            bev_pos_1 = None
            multiscale_inputs = None
        else:
            assert self.use_lidar
            batch_size = pts_inputs[0].shape[0]
            # preprocess LiDAR features, following FocalFormer3D focal_decoder.py
            lidar_feat = pts_inputs[0]
            extra_feats = pts_inputs[1].pop(-1)
            lidar_feat_flatten = lidar_feat.view(
                batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
            
            bev_pos_1 = self.bev_pos_1.repeat(batch_size, 1, 1).to(lidar_feat.device)
            bev_pos_2 = self.bev_pos_2.repeat(batch_size, 1, 1).to(lidar_feat.device)
            bev_pos_4 = self.bev_pos_4.repeat(batch_size, 1, 1).to(lidar_feat.device)
            bev_pos = torch.cat([bev_pos_1, bev_pos_2, bev_pos_4], dim=1)

            multiscale_inputs = [extra_feats]
            multiscale_inputs.append(self.dconv(multiscale_inputs[-1]))
            multiscale_inputs.append(
                self.dconv2(multiscale_inputs[-1]))
            multiscale_inputs_flatten = torch.cat(
                [i.flatten(2, 3) for i in multiscale_inputs], dim=-1)
            # moved the line below out of the decoder layer for loop in focal_decoder

            ################## Deformable Parameters ##################
            # always do multiscale
            spatial_shapes = torch.as_tensor(
                [i.shape[2:] for i in multiscale_inputs], 
                device=multiscale_inputs_flatten.device)
            level_start_index = torch.as_tensor(
                [0, *(torch.cumsum(torch.prod(spatial_shapes, dim=1), dim=0)[:-1])], 
                device=multiscale_inputs_flatten.device)

            # lidar feat
            lidar_feat_flatten = multiscale_inputs_flatten
            MSDA_kwargs = dict(
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=torch.ones((batch_size, 1, 2), device='cuda'),
            )
            ################## Deformable Parameters ##################

            ################## bev_pos_embed ##################
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
            ################## bev_pos_embed ##################

            multistage_feats = pts_inputs[1]
            multistage_feats.insert(0, lidar_feat)

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
            dense_heatmap_list,
            multistage_acc_masks,
        ) = self.instance_bank.get(
            batch_size,
            timestamp,
            batched_global2lidar=batched_global2lidar,
            # pass dn_metas to instance_bank.get to do forward timestep update
            dn_metas=self.sampler.dn_metas,
            multistage_feats=multistage_feats,
            multiscale_lidar_feats=multiscale_inputs,
            bev_pos=bev_pos_1
        )
        num_learned_grp = anchor.shape[1]
        if self.instance_bank.heatmap_init:
            output_dict = dict(
                heatmap_bboxes=anchor,
                dense_heatmap_list=dense_heatmap_list,
                multistage_acc_masks=multistage_acc_masks
            )
        else:
            output_dict = dict()

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            dn_metas = self.sampler.get_dn_anchors(
                cls_target=[ds.gt_instances_3d.labels_3d for ds in batch_data_samples],
                box_target=[ds.gt_instances_3d.bboxes_3d for ds in batch_data_samples],
                multiscale_lidar_feats=multiscale_inputs,
                num_bbox_pool_points=self.instance_bank.num_bbox_pool_points,
                roi_mlp=self.instance_bank.roi_mlp,
                cat_encoding=self.instance_bank.cat_encoding,
                gt_instance_inds=[ds.gt_instances_3d.instance_inds for ds in batch_data_samples],
            )
        if dn_metas is not None:
            # TODO put this condition into a separate method
            (
                dn_anchor,  # (bs, num_dn_groups, num_dn_anchor, 11)
                dn_feat, # (bs, num_dn_groups, num_dn_anchor, embbed_dim)
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_groups = dn_anchor.shape[1]
            dn_per_dn_grp = dn_anchor.shape[2]  # num dn in a single dn group
            # check anchor dimension. If they don't match, pad dn_anchor with zeros
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_groups, dn_per_dn_grp, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            assert num_dn_groups % num_learned_grp == 0  # only support equal division
            dn_grp_per_learned_grp = num_dn_groups // num_learned_grp
            dn_per_lrn_grp = dn_per_dn_grp * dn_grp_per_learned_grp
            # concat along the query dim
            if num_learned_grp != num_dn_groups:
                # divide dn groups evenly among learned groups
                dn_anchor = dn_anchor.view(
                    batch_size, num_learned_grp, dn_per_lrn_grp, -1
                ) # (bs, num_learned_grp, num_instance, embbed_dim)
                dn_feat = dn_feat.view(
                    batch_size, num_learned_grp, dn_per_lrn_grp, -1
                )
            # (bs, num_learned_grp, num_instance, embbed_dim)
            anchor = torch.cat([anchor, dn_anchor], dim=2)

            instance_feature = torch.cat([instance_feature, dn_feat], dim=2,)

            # construct attn mask
            num_instance = instance_feature.shape[2]
            num_free_instance = num_instance - dn_per_lrn_grp
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            # mask false means attend, so here we attend to the free_instances
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask[:dn_per_lrn_grp, :dn_per_lrn_grp]

        anchor_embed = self.anchor_encoder(anchor)

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                # flatten things along the batch and group dims
                instance_feature_flattened = instance_feature.flatten(0, 1)
                anchor_embed_flattened = anchor_embed.flatten(0, 1)
                # attend to learnable instances (300) + temp instances (600) or in first frame case only learnable instances (900)
                # in temp_gnn, do not attend to dn instances
                instance_feature_flattened = self.graph_model(
                    index=i,
                    query=instance_feature_flattened,
                    key=instance_feature_flattened[:, :self.instance_bank.num_anchor],
                    value=instance_feature_flattened[:, :self.instance_bank.num_anchor],
                    query_pos=anchor_embed_flattened,
                    key_pos=anchor_embed_flattened[:, :self.instance_bank.num_anchor],
                )
                # reshape back to batch x num_learned_grp x num_instance x embed_dim
                instance_feature = instance_feature_flattened.view(
                    batch_size, num_learned_grp, -1, instance_feature.shape[-1])
            elif op == "gnn":
                instance_feature_flattened = instance_feature.flatten(0, 1)
                anchor_embed_flattened = anchor_embed.flatten(0, 1)
                instance_feature = self.graph_model(
                    index=i,
                    query=instance_feature_flattened,
                    value=instance_feature_flattened,
                    query_pos=anchor_embed_flattened,
                    attn_mask=attn_mask,
                )
                # reshape back to batch x num_learned_grp x num_instance x embed_dim
                instance_feature = instance_feature.view(
                    batch_size, num_learned_grp, -1, instance_feature.shape[-1])
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                # flatten over the num groups and num queries
                instance_feature_flattened = self.layers[i](
                    instance_feature.flatten(1, 2),
                    anchor.flatten(1, 2),
                    anchor_embed.flatten(1, 2),
                    feature_maps,
                    projection_mat,
                    image_wh,
                )
                # reshape back to batch x num_learned_grp x num_instance x embed_dim
                instance_feature = instance_feature_flattened.view(
                    batch_size, num_learned_grp, -1, instance_feature.shape[-1])
            elif op == "deformable_lidar":
                # flatten along the num groups and num queries
                instance_feature_flattened = instance_feature.flatten(1, 2)
                anchor_flattened = anchor.flatten(1, 2)
                anchor_embed_flattened = anchor_embed.flatten(1, 2)
                # normalize anchor to [0, 1] to get reference_points
                reference_points = anchor_flattened[..., :2]
                reference_points = (reference_points - self.point_cloud_range[:2].to(
                    reference_points.device)) / self.ref_point_norm[:2].to(reference_points.device)
                reference_points = reference_points.clamp(0, 1)

                # expand reference_points to match the shape of valid ratios, see mmdet DeformableDetrTransformerDecoder
                reference_points_input = \
                    reference_points[:, :, None] * \
                    MSDA_kwargs["valid_ratios"][:, None]

                deform_out_flattened = self.layers[i](
                    query=instance_feature_flattened,  # B x N x C
                    value=pos_lidar_feat_flatten.permute(
                        0, 2, 1),  # B C Pv -> B Pv C
                    key=pos_lidar_feat_flatten.permute(  # only used by dense attn
                        0, 2, 1),  # B C Pv -> B Pv C
                    query_pos=anchor_embed_flattened,  # B N C
                    reference_points=reference_points_input,  # B N num_levels 2
                    **MSDA_kwargs,)

                # reshape back to batch x num_learned_grp x num_instance x embed_dim
                deformable_output = deform_out_flattened.view(
                    batch_size, num_learned_grp, -1, instance_feature.shape[-1])

                # follow DeformableFeatureAggregation, concat the output
                instance_feature = torch.cat(
                    [instance_feature, deformable_output], dim=-1)

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
                    # update learned queries with cached TQ
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    # update DN queries with cached temp DN group
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

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            # classification, prediction, quality are length of total decoder layers
            dn_classification = [
                x[:, :, num_free_instance:] for x in classification
            ]
            classification = [x[:, :, :num_free_instance] for x in classification]
            dn_prediction = [x[:, :, num_free_instance:] for x in prediction]
            prediction = [x[:, :, :num_free_instance] for x in prediction]
            quality = [
                x[:, :, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output_dict.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output_dict.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask

            # cache dn_metas for temporal denoising
            dn_instance_feature = instance_feature[:, :, num_free_instance:]
            dn_anchor = anchor[:, :, num_free_instance:]
            # reshape back from (bs, num_learned_groups, num_dn_anchor * dn_grp_per_lrn_gro, embed_dim) 
            # to (bs, num_dn_groups, num_dn_anchor, embbed_dim)
            dn_instance_feature = dn_instance_feature.view(
                batch_size, num_dn_groups, dn_per_dn_grp, -1)
            dn_anchor = dn_anchor.view(
                batch_size, num_dn_groups, dn_per_dn_grp, -1)
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

            # split off learned queries for caching
            instance_feature = instance_feature[:, :, :num_free_instance]
            anchor = anchor[:, :, :num_free_instance]
            cls = cls[:, :, :num_free_instance]

        output_dict.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        if not self.training:
            # assign instance_inds to all predictions for inference
            instance_inds = self.instance_bank.get_instance_ind(
                cls, self.decoder.score_threshold
            )
            output_dict["instance_inds"] = instance_inds
        else:
            output_dict["instance_inds"] = None

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature,
            anchor,
            cls,
            timestamp,
            [x["lidar2global"] for x in batch_metas],
            output_dict["instance_inds"]
        )
        return output_dict

    def loss(self, model_outs, batch_data_samples):
        gt_cls = [bs.gt_instances_3d.labels_3d for bs in batch_data_samples]
        gt_reg = [bs.gt_instances_3d.bboxes_3d for bs in batch_data_samples]
        gt_id = [bs.gt_instances_3d.instance_inds for bs in batch_data_samples]
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output_dict = {}
        prev_instance_inds = self.instance_bank.instance_inds_training
        batch_size = len(gt_cls)
        if prev_instance_inds is None:
            prev_instance_inds = [None for i in range(batch_size)]
        else:
            # if not mask (not the same sequence), set to None
            prev_instance_inds = [
                prev_instance_inds[bs] if self.instance_bank.mask[bs] else None
                for bs in range(batch_size)
            ]
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            # TODO move code to compute loss on a given decoder layer to a separate method
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights, id_target = self.sampler.sample(
                cls,
                reg,
                gt_cls,
                gt_reg,
                gt_id,
                # only use prev_instance_inds if self.sampler.supervise_qc and not single_frame_decoder output
                prev_instance_inds if decoder_idx >= self.num_single_frame_decoder else None,
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)).item(),
                1.0
            )
            confidences = cls.max(dim=-1).values.sigmoid()
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, confidences > threshold
                )

            # flatten batch, num group, num query dims
            cls_target = cls_target.flatten()
            cls_loss = self.loss_cls(
                cls.flatten(end_dim=2),
                cls_target,
                avg_factor=num_pos)

            # mask determines which elements to consider for regression loss
            mask = mask.flatten()
            reg_weights *= self.reg_weights.to(reg_weights.device)

            reg_target = reg_target.flatten(end_dim=2)[mask]
            reg = reg.flatten(end_dim=2)[mask]
            reg_weights = reg_weights.flatten(end_dim=2)[mask]

            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=2)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output_dict[f"loss_cls_{decoder_idx}"] = cls_loss
            output_dict.update(reg_loss)

            # compute metrics for query consistency
            qc_metrics = self.compute_qc_metrics(
                gt_id, 
                confidences, 
                id_target, 
                prev_instance_inds if decoder_idx >= self.num_single_frame_decoder else None,
            )
            for key, val in qc_metrics.items():
                if not val.isnan():
                    # add decoder suffix to qc metrics
                    output_dict[f"qc_metrics/{key}_{decoder_idx}"] = val

        # cache the id_target of the final layer for the next timestep
        # assuming non-zero decoder layers
        bs, num_groups, k = self.instance_bank.cached_indices.shape
        batch_indices = torch.arange(bs).view(
            -1, 1, 1).expand(-1, num_groups, k)

        group_indices = torch.arange(num_groups).view(
            1, -1, 1).expand(bs, -1, k)
        # cache id target to intsance_inds for the next timestep
        self.instance_bank.instance_inds_training = id_target[
            batch_indices, group_indices,
            self.instance_bank.cached_indices]

        # compute losses on dn queries
        if "dn_prediction" in model_outs:
            dn_losses = self.compute_dn_losses(model_outs)
            output_dict.update(dn_losses)

        # compute losses on dense_heatmap_list
        if model_outs.get('dense_heatmap_list', None) is not None:
            heatmap_losses = self.compute_heatmap_losses(
                model_outs['dense_heatmap_list'], 
                model_outs['multistage_acc_masks'],
                model_outs['heatmap_bboxes'],
                batch_data_samples)
            output_dict.update(heatmap_losses)

        return output_dict

    def compute_dn_losses(self, model_outs):
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

        output_dict = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                """
                Update variables with temp variants
                """
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=2)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=2)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output_dict[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output_dict.update(reg_loss)
        return output_dict

    def compute_qc_metrics(self, gt_id, conf, id_target, prev_instance_inds=None):
        """
        Compute query consistency metrics
        Averaged over all objects in all samples in the batch (weighted more towards samples with more objects)
        TODO currently does not correctly split tq/pq for multi-frame decoder layers (decoder layer > 1) and first frame of sequence.
        In this scenario, prev_instance_inds = [None] *batch_size, and all the preds are technically pq.
        Current behaviour is to still split the preds assuming there are both tq and pq present
        """
        if prev_instance_inds is None or all([x is None for x in prev_instance_inds]):
            num_temp_instances = 0
        else:
            num_temp_instances = self.instance_bank.num_temp_instances

        batch_size = len(gt_id)
        tq_conf, pq_conf = conf[:, :, :num_temp_instances], conf[:,:, num_temp_instances:]
        tq_id_target, pq_id_target = id_target[:, :, :num_temp_instances], id_target[:, :, num_temp_instances:]
        device = pq_conf.device

        if prev_instance_inds is None or all([x is None for x in prev_instance_inds]):
            valid_prev_instance_inds = [torch.empty(
                (0,), device=conf.device) for _ in range(batch_size)]
        else:
            # Convert instance_inds to a tensor and filter out UNTRACKED_ID values
            valid_prev_instance_inds = [
                inds[inds != UNTRACKED_ID] if inds is not None else torch.empty(
                    (0,), device=device)
                for inds in prev_instance_inds
            ]

        # Create a mask for which pq were in prev frame, (bs, num_temp_instances)
        prev_pq_mask = torch.stack([torch.isin(pq_id_target_i, valid_prev_instance_inds_i) for (
            pq_id_target_i, valid_prev_instance_inds_i) in zip(pq_id_target, valid_prev_instance_inds)])
        # Create a mask for current IDs, check not UNTRACKED_ID
        pos_pq_mask = pq_id_target != UNTRACKED_ID
        # Newborn mask is true where the pq pred is pos in curr but not in prev_mask
        newborn_mask = pos_pq_mask & ~prev_pq_mask
        # pq_tp is true when the pq is assigned (!=UNTRACKED_iD) and it is a newborn gt (not in prev frame)
        pq_tp = newborn_mask.sum()
        # pq_fp: pq assigned but it is not a newborn gt (it was a prev tracked obj)
        pq_fp = pos_pq_mask.sum() - pq_tp

        # pq_fn: a newborn gt that was assigned to a tq, not a hinderance to query consistency, ignore
        metric_dict = dict(
            pq_tp_conf=pq_conf[newborn_mask].nanmean(),
            pq_fp_conf=pq_conf[pos_pq_mask & prev_pq_mask].nanmean(),
            pq_neg_conf=pq_conf[~pos_pq_mask].nanmean(),
            # of the total pos pq predictions, how many were actual newborn obj
            pq_precision=pq_tp / (pq_tp + pq_fp) if (pq_tp + pq_fp) > 0 else torch.tensor(0.0),
        )

        # redundant condition, but keeping for clarity
        if num_temp_instances > 0 and prev_instance_inds is not None and not all([x is None for x in prev_instance_inds]):
            pos_tq_mask = tq_id_target != UNTRACKED_ID
            # tq_tp: tq assigned (!=-1) and it was the same previously tracked obj
            tq_tp_mask = torch.stack([
                torch.zeros_like(tq_id_target_i, dtype=torch.bool)
                if prev_instance_inds_i is None else
                (tq_id_target_i == prev_instance_inds_i) & pos_tq_mask_i
                for (tq_id_target_i, prev_instance_inds_i, pos_tq_mask_i)
                in zip(tq_id_target, prev_instance_inds, pos_tq_mask)
            ])
            tq_tp = tq_tp_mask.sum()

            tq_match_mask = torch.stack([
                torch.ones_like(tq_id_target_i, dtype=torch.bool) 
                if prev_instance_inds_i is None else
                tq_id_target_i == prev_instance_inds_i  # tq_id_target_i matches prev_instance_inds_i
                for (tq_id_target_i, prev_instance_inds_i)
                in zip(tq_id_target, prev_instance_inds) # iterate over batch
            ])
            # tq_fp: tq assigned (!=UNTRACKED_ID) and it was a different gt
            # handle the case of prev_instance_ind[i] being none
            tq_fp_mask = pos_tq_mask & (~tq_match_mask)
            tq_fp = tq_fp_mask.sum()

            carryover_mask = torch.stack([
                torch.zeros_like(tq_id_target_i, dtype=torch.bool)
                if prev_instance_inds_i is None else
                torch.isin(prev_instance_inds_i, gt_id_i) # object was in prev frame and is in current frame
                for (tq_id_target_i, prev_instance_inds_i, gt_id_i)
                in zip(tq_id_target, prev_instance_inds, gt_id) # iterate over samples in batch
            ])
            # tq_fn: prev_inst is in curr frame but corresponding query is not a tp
            tq_fn_mask = carryover_mask & ~tq_tp_mask
            tq_fn = tq_fn_mask.sum()

            metric_dict.update(
                tq_tp_conf=tq_conf[tq_tp_mask].nanmean(),
                tq_fp_conf=tq_conf[tq_fp_mask].nanmean(),
                tq_fn_conf=tq_conf[tq_fn_mask].nanmean(),
                # of the total pos tq predictions, how many were actual prev tracked obj
                tq_precision=tq_tp / (tq_tp + tq_fp) if tq_tp + \
                tq_fp > 0 else torch.tensor(0.0, device=device),
                # of the total tracked obj that are also in current frame, how many maintained query consistency
                tq_recall=tq_tp / (tq_tp + tq_fn) if tq_tp + \
                tq_fn > 0 else torch.tensor(0.0, device=device),
            )

        return metric_dict

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        """
        Prepare the outputs for dn loss calculation
        Flatten along batch and other dims, filter out invalid instances, and get the number of positive instances
        """
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(
            end_dim=2)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=2
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=2
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = (dn_cls_target != NEG_DN_CLS_TARGET) & (
            dn_cls_target != PAD_CLS_TARGET)
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = self.reg_weights[None].tile(
            dn_reg_target.shape[0], 1
        ).to(dn_reg_target.device)
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask, dtype=reg_weights.dtype)).item(),
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

    def compute_heatmap_losses(self, dense_heatmap, multistage_acc_masks, heatmap_bboxes, batch_data_samples):
        output_dict = dict()
        # dense_heatmap = torch.stack(dense_heatmap, dim=0) # (num heatmap stages, bs, num classes, H, W)
        # multistage_acc_masks = torch.stack(multistage_acc_masks, dim=0) # (num heatmap stages, bs, num classes, H, W)
        # compute heatmap targets
        # truncate heatmap_bboxes to len(self.reg_weights)
        heatmap_bboxes = heatmap_bboxes[..., : len(self.reg_weights)]
        gt_heatmap, gt_heatmap_bboxes, reg_weight = self.sampler.get_heatmap_target(
            dense_heatmap[0, 0],  # only used to get shape, only need to pass from 1 stage, 1 group
            heatmap_bboxes, 
            batch_data_samples)
        reg_weight *= self.reg_weights.to(reg_weight.device)
        # mask out gt that should be ignored (why is this necessary? following ff3d, but could be incorrect)
        # if we pass masks as weight, why do we need to mask out the gt target?
        # repeat gt_heatmap to match number of stages
        gt_heatmap = gt_heatmap.repeat(
            self.instance_bank.num_heatmap_stages, self.instance_bank.num_learned_groups, 1, 1, 1, 1) * multistage_acc_masks

        # compute num_pos based on total gt bboxes
        num_pos = max(1, gt_heatmap.eq(1).float().sum().item())
        # compute heatmap loss
        output_dict['loss_heatmap'] = self.loss_heatmap(
            clip_sigmoid(dense_heatmap),
            gt_heatmap,
            weight=multistage_acc_masks,
            avg_factor=num_pos
        )
        mask = torch.logical_not(torch.all(gt_heatmap_bboxes == 0, dim=-1))
        num_pos = max(
            reduce_mean(torch.sum(mask).to(dtype=heatmap_bboxes.dtype)).item(),
            1.0
        )
        # compute heatmap bbox loss
        output_dict['loss_heatmap_bbox'] = self.loss_heatmap_reg(
            heatmap_bboxes,
            gt_heatmap_bboxes,
            weight=reg_weight,
            avg_factor=num_pos
        )
        return output_dict

    def post_process(self, model_outs, output_idx=-1):
        # squeeze on the group dim
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_inds"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
