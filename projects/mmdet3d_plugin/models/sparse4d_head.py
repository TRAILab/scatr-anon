# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import points_in_boxes_part
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
        point_cloud_range: List[float],
        modality: str = "camera",
        multistage_heatmap: Union[int, bool] = False,
        bevpos: bool = True,
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

        # focalformer
        self.use_lidar = modality == "lidar"
        self.use_camera = modality == "camera"
        self.multistage_heatmap = multistage_heatmap
        if self.use_lidar:
            self.pos_embed_learned = MLP(
                128 * 5, self.embed_dims, self.embed_dims, 2)
            # X-min, Y-min, Z-min, X-max, Y-max, Z-max
            # used for normalizing anchor to be [0, 1] for reference_points
            self.point_cloud_range = torch.nn.Parameter(
                torch.tensor(point_cloud_range), requires_grad=False)
            self.ref_point_norm = self.point_cloud_range[3:] - \
                self.point_cloud_range[:3]
            self.bevpos = bevpos

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
            if self.multiscale:
                bev_pos_2 = self.create_2D_grid(
                    lidar_feat.shape[2] // 2, lidar_feat.shape[2] // 2).repeat(batch_size, 1, 1).to(lidar_feat.device) * 2
                bev_pos_4 = self.create_2D_grid(
                    lidar_feat.shape[2] // 4, lidar_feat.shape[2] // 4).repeat(batch_size, 1, 1).to(lidar_feat.device) * 4
            dense_heatmap_boxes = None
            query_box = None

        # TODO move the following code to a separate function
        if not self.multistage_heatmap and self.use_lidar:
            dense_heatmap = self.heatmap_head(lidar_feat)
            if self.input_img or self.iterbev_wo_img:
                if isinstance(pts_inputs[1], (list, tuple)):
                    new_lidar_feat = pts_inputs[1][-1]
                else:
                    new_lidar_feat = pts_inputs[1]
                lidar_feat_flatten = new_lidar_feat.view(
                    *lidar_feat_flatten.shape)

                dense_heatmap_img = self.heatmap_head_img(
                    new_lidar_feat.view(lidar_feat.shape))  # [BS, num_classes, H, W]
                heatmap = (dense_heatmap.detach().sigmoid() +
                           dense_heatmap_img.detach().sigmoid()) / 2
            else:
                heatmap = dense_heatmap.detach().sigmoid()
                new_lidar_feat = lidar_feat
            if self.input_img or self.iterbev_wo_img:
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
        elif self.use_lidar:
            dense_heatmap = self.heatmap_head(lidar_feat)  # original

            multistage_feats = pts_inputs[1]
            if self.reuse_first_heatmap:
                multistage_feats.insert(0, lidar_feat)

            query_labels = []
            query_feats = []
            query_boxes = []
            query_poses = []
            query_heatmap_scores = []
            acc_masks = torch.ones_like(dense_heatmap).view(batch_size, -1)
            multistage_masks = []
            multistage_masks_independent_visualize = []
            heatmap_train = []
            multistage_bev_preds = []
            for i in range(self.multistage_heatmap):
                if i == 0 and self.reuse_first_heatmap:
                    if self.heatmap_box:
                        assert self.test_cfg['dataset'] == 'nuScenes'
                        shared_feat = multistage_feats[i]
                        dense_preds = []
                        dense_heatmap_boxes = []
                        if not self.thin_heatmap_box:
                            for task_id, task in enumerate(self.multi_stage_task_heads[i]):
                                dense_preds.append(task(shared_feat))
                                dense_pred = dense_preds[-1]
                                if 'vel' in dense_pred:
                                    dense_pred = (
                                        dense_pred['reg'], dense_pred['height'], dense_pred['dim'], dense_pred['rot'], dense_pred['vel'])
                                else:
                                    dense_pred = (
                                        dense_pred['reg'], dense_pred['height'], dense_pred['dim'], dense_pred['rot'])
                                dense_pred = torch.cat(dense_pred, dim=1)[
                                    :, :, None].expand(-1, -1, self.heatmap_tasks[task_id]['num_class'], -1, -1)
                                dense_heatmap_boxes.append(dense_pred)
                        else:
                            dense_heatmap_boxes_raw = self.multi_stage_task_heads[i](
                                shared_feat)
                            dense_preds_raw = torch.split(
                                dense_heatmap_boxes_raw, [10] * 6, dim=1)
                            for task_id in range(len(self.heatmap_tasks)):
                                dense_pred = torch.split(dense_preds_raw[task_id], [
                                                         2, 1, 3, 2, 2], dim=1)
                                dense_preds.append(dict(
                                    reg=dense_pred[0], height=dense_pred[1], dim=dense_pred[2], rot=dense_pred[3], vel=dense_pred[4]))
                                dense_heatmap_boxes.append(dense_preds_raw[task_id][:, :, None].expand(
                                    -1, -1, self.heatmap_tasks[task_id]['num_class'], -1, -1))
                        multistage_bev_preds.append(dense_preds)
                        dense_heatmap_boxes = torch.cat(
                            dense_heatmap_boxes, dim=2)

                    heatmap = dense_heatmap.detach().sigmoid()
                    heatmap_train.append(dense_heatmap)
                    multistage_masks.append(
                        acc_masks.view(*heatmap.shape).clone())
                    # remove early positive
                    heatmap = heatmap * acc_masks.view(*heatmap.shape)
                else:
                    if not self.heatmap_box:
                        dense_heatmap_img = self.heatmap_head_img[i](
                            multistage_feats[i])
                    else:
                        assert self.test_cfg['dataset'] == 'nuScenes'
                        shared_feat = multistage_feats[i]
                        dense_preds = []
                        dense_heatmap_boxes = []
                        if not self.thin_heatmap_box:
                            for task_id, task in enumerate(self.multi_stage_task_heads[i]):
                                dense_preds.append(task(shared_feat))
                                dense_pred = dense_preds[-1]
                                dense_pred = torch.cat((dense_pred['reg'], dense_pred['height'], dense_pred['dim'],
                                                        dense_pred['rot'], dense_pred['vel']), dim=1)[:, :, None].expand(-1, -1, self.heatmap_tasks[task_id]['num_class'], -1, -1)
                                dense_heatmap_boxes.append(dense_pred)
                            dense_heatmap_img = torch.cat(
                                [p['heatmap'] for p in dense_preds], dim=1)
                        else:
                            dense_heatmap_boxes_raw = self.multi_stage_task_heads[i](
                                shared_feat)
                            dense_preds_raw = torch.split(
                                dense_heatmap_boxes_raw, [10] * 6, dim=1)
                            for task_id in range(len(self.heatmap_tasks)):
                                dense_pred = torch.split(dense_preds_raw[task_id], [
                                                         2, 1, 3, 2, 2], dim=1)
                                dense_preds.append(dict(
                                    reg=dense_pred[0], height=dense_pred[1], dim=dense_pred[2], rot=dense_pred[3], vel=dense_pred[4]))
                                dense_heatmap_boxes.append(dense_preds_raw[task_id][:, :, None].expand(
                                    -1, -1, self.heatmap_tasks[task_id]['num_class'], -1, -1))
                            dense_heatmap_img = self.heatmap_head_img[i](
                                multistage_feats[i])
                        multistage_bev_preds.append(dense_preds)
                        dense_heatmap_boxes = torch.cat(
                            dense_heatmap_boxes, dim=2)

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

                if self.heatmap_box:
                    box_dim = dense_heatmap_boxes.shape[1]
                    dense_heatmap_boxes = dense_heatmap_boxes.detach().view(
                        batch_size, box_dim, self.num_classes, heatmap.shape[-1])
                    assert self.test_cfg['dataset'] == 'nuScenes'
                    # learns from center_int to target offsets
                    dense_heatmap_boxes[:, :2, :, :] += bev_pos.int().float().transpose(
                        1, 2)[:, :, None].expand_as(dense_heatmap_boxes[:, :2, :, :])
                    dense_heatmap_boxes[:, 2:3, :, :] = dense_heatmap_boxes[:, 2:3, :, :].clip(
                        min=-5., max=3.)  # gravi center
                    dense_heatmap_boxes[:, 3:6, :, :] = dense_heatmap_boxes[:, 3:6, :, :].clip(
                        min=np.log(0.5), max=np.log(15))  # box dim log
                    dense_heatmap_boxes[:, 6:8, :, :] = dense_heatmap_boxes[:, 6:8, :, :].clip(
                        min=-1., max=1.)  # sincos
                    dense_heatmap_boxes[:, 8:10, :, :] = dense_heatmap_boxes[:, 8:10, :, :].clip(
                        min=-15., max=15.)

                    dense_heatmap_boxes = dense_heatmap_boxes.view(
                        batch_size, box_dim, self.num_classes*heatmap.shape[-1])

                    query_box = dense_heatmap_boxes.gather(
                        index=top_proposals[:, None, :].expand(-1, box_dim, -1), dim=-1)
                    query_boxes.append(query_box)

                ################ select to ignore ######################
                if self.mask_heatmap_mode == 'pos':
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes, heatmap.shape[-1])
                    selected_mask.scatter_(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=2,
                                           src=acc_masks.new_ones((batch_size, self.num_classes, heatmap.shape[-1])))
                elif self.mask_heatmap_mode == 'poscls':
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes * heatmap.shape[-1])
                    selected_mask.scatter_(index=top_proposals, dim=1, src=torch.ones_like(
                        top_proposals, dtype=acc_masks.dtype))
                elif self.mask_heatmap_mode == 'boxcls':
                    boxmask_margin = 1.
                    boxmask_margin_ratio = None

                    assert self.test_cfg['dataset'] == 'nuScenes'
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes * heatmap.shape[-1])
                    selected_mask.scatter_(index=top_proposals, dim=1, src=torch.ones_like(
                        top_proposals, dtype=acc_masks.dtype))

                    # bev_dim > 108 / 180 = 0.6
                    def pos_inside_boxes(query_box, bev_pos, margin, min_bev_dim, margin_ratio=None):
                        assert query_box.shape[1] >= 9
                        rot, dim, center, height, vel = query_box[:, 6:8], query_box[:,
                                                                                     3:6], query_box[:, 0:2], query_box[:, 2:3], query_box[:, 8:]
                        query_boxes_std = self.bbox_coder.decode_box(
                            rot.clone(), dim.clone(), center.clone(), height.clone(), vel.clone())
                        pc_range = torch.as_tensor(
                            [-54, -54, -5.0, 54, 54, 3.0], device='cuda')
                        query_boxes_std[..., [0,]] = query_boxes_std[..., [
                            0,]].clip(min=pc_range[0], max=pc_range[3])
                        query_boxes_std[..., [1,]] = query_boxes_std[..., [
                            1,]].clip(min=pc_range[1], max=pc_range[4])
                        if margin_ratio is not None and margin_ratio > 0.:
                            query_boxes_std[..., [3, 4]] *= (1. - margin_ratio)
                        else:
                            query_boxes_std[..., [3, 4]] -= margin
                        query_boxes_std[..., [3, 4]] = query_boxes_std[..., [
                            3, 4]].clip(min=min_bev_dim, max=10.)
                        query_boxes_std[..., 5] = 1000  # height -> max
                        query_boxes_std[..., 2] = -100.  # bottom center
                        temp_bev_pos = self.bbox_coder.decode_center(
                            bev_pos.transpose(1, 2))  # bev points
                        temp_bev_pos = torch.cat([temp_bev_pos, temp_bev_pos.new_zeros(
                            batch_size, 1, bev_pos.shape[1])], dim=1).transpose(1, 2)
                        inside_boxes = points_in_boxes_part(
                            temp_bev_pos,
                            query_boxes_std[:, :, :7])

                        return inside_boxes

                    inside_boxes = pos_inside_boxes(
                        query_box, bev_pos, margin=boxmask_margin, min_bev_dim=0.7, margin_ratio=boxmask_margin_ratio)
                    bev_pos_class = top_proposals_class.gather(
                        index=inside_boxes.clip(min=0).long(), dim=1)
                    bev_pos_class[inside_boxes == -
                                  1] = self.num_classes  # background
                    selected_mask_box = acc_masks.new_zeros(
                        batch_size, self.num_classes + 1, heatmap.shape[-1])
                    selected_mask_box.scatter_(index=bev_pos_class[:, None], dim=1, src=torch.ones_like(
                        bev_pos_class[:, None], dtype=acc_masks.dtype))
                    selected_mask_box = selected_mask_box[:, :self.num_classes].reshape(
                        batch_size, self.num_classes * heatmap.shape[-1])

                    selected_mask = (
                        selected_mask + selected_mask_box > 0.1).float()
                else:
                    selected_mask = acc_masks.new_zeros(
                        batch_size, self.num_classes * heatmap.shape[-1])

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
            if self.heatmap_box:
                query_box = torch.cat(query_boxes, dim=2)

            self.num_proposals = self.num_proposals_ori * self.multistage_heatmap

        if self.use_lidar:
            if self.training:
                self.num_gts = [i.shape[0] for i in gt_labels_3d]
                self.max_num_gts = max(self.num_gts)
            query_labels = self.query_labels

            # skip focal former DN generation

            if self.multiscale:
                if not self.multistage_heatmap:
                    lidar_feat = new_lidar_feat
                else:
                    if self.extra_feat:
                        lidar_feat = extra_feats
                    else:
                        lidar_feat = multistage_feats[-1]

                multiscale_inputs = [lidar_feat]
                if self.multiscale:
                    multiscale_inputs.append(self.dconv(multiscale_inputs[-1]))
                    multiscale_inputs.append(
                        self.dconv2(multiscale_inputs[-1]))
                multiscale_inputs_flatten = torch.cat(
                    [i.flatten(2, 3) for i in multiscale_inputs], dim=-1)
                # moved the line below out of the decoder layer for loop in focal_decoder
                bev_pos = torch.cat([bev_pos, bev_pos_2, bev_pos_4], dim=1)

            ################## Deformable Parameters #############
            if not self.multiscale:
                W, H = lidar_feat.shape[-2:]
                spatial_shapes = torch.as_tensor(
                    [[W, H]], dtype=torch.long, device='cuda')
                level_start_index = torch.as_tensor(
                    [0,], dtype=torch.long, device='cuda')
            else:
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
            breakpoint()  # verify MSDA kwargs

            if self.bevpos:
                bev_reference_points = bev_pos / \
                    torch.flip(spatial_shapes[:1], dims=(1,))[:, None]
                bev_sine_pos = gen_sineembed_for_position(
                    bev_reference_points[:, :, :2])
                breakpoint()  # check size of bev_sin_pos, what is channel size? pos_embed_learned needs to match
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
                breakpoint()  # verify input shapes are as expected
                # normalize anchor to [0, 1] to get reference_points
                reference_points = anchor[..., :2]
                reference_points = (reference_points - torch.tensor(self.point_cloud_range[:2], device=reference_points.device)) / torch.tensor(
                    self.ref_point_norm[:2], device=reference_points.device)

                # verify reference points are in correct range [0,1]
                breakpoint()
                instance_feature = self.layers[i](
                    query=instance_feature,  # B x N x C
                    value=pos_lidar_feat_flatten.permute(
                        0, 2, 1),  # B C Pv -> B Pv C
                    query_pos=anchor_embed,  # B 600 C
                    reference_points=reference_points,  # B 600 2
                    **MSDA_kwargs,)
                breakpoint()  # verify output still has the same shape
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

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature,
            anchor,
            cls,
            timestamp,
            [x["lidar2global"] for x in batch_metas],
            feature_maps
        )
        if not self.training:
            instance_inds = self.instance_bank.get_instance_ind(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_inds"] = instance_inds
        return output

    def loss(self, model_outs, batch_data_samples):
        gt_cls = [bs.gt_instances_3d.labels_3d for bs in batch_data_samples]
        gt_reg = [bs.gt_instances_3d.bboxes_3d for bs in batch_data_samples]
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                gt_cls,
                gt_reg,
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

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
