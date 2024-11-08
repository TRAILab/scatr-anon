# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature
from typing import Dict, List, Optional

import numpy as np
import torch
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from torch import Tensor

from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["Sparse4D"]


@MODELS.register_module()
class Sparse4D(MVXTwoStageDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            raise NotImplementedError("not pretrained is not supported.")
            backbone.pretrained = pretrained
        self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        self.head = MODELS.build(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = MODELS.build(depth_branch)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    def extract_feat(self, img, return_depth: bool = False, focal=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            raise NotImplementedError("metas is not supported.")
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, focal)
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        return feature_maps, depths

    def loss(self, batch_inputs_dict: Dict,
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_img = batch_inputs_dict["img"]
        batch_focal = torch.tensor([
            [intr[0, 0] for intr in bs.metainfo["intrinsics"]]
            for bs in batch_data_samples], device=batch_img.device)  # (bs, 6)
        feature_maps, depths = self.extract_feat(batch_img, True, batch_focal)
        # timestamp needs to be type double to avoid quantization errors
        timestamp = torch.tensor([bs.metainfo["timestamp"]
                                 for bs in batch_data_samples], dtype=torch.float64)
        model_outs = self.head(
            feature_maps,
            timestamp=timestamp,
            projection_mat=batch_inputs_dict["lidar2img"].to(torch.float32),
            # flip (H, W) to (W, H)
            image_wh=batch_inputs_dict["img_shape"][..., [1, 0]],
            batch_data_samples=batch_data_samples,
        )
        gt_cls = [bs.gt_instances_3d.labels_3d for bs in batch_data_samples]
        gt_reg = [bs.gt_instances_3d.bboxes_3d for bs in batch_data_samples]

        output = self.head.loss(
            model_outs,
            gt_cls,
            gt_reg)
        gt_depth = [
            torch.from_numpy(
                np.stack([depth.metainfo["gt_depth"][i]
                         for depth in batch_data_samples])
            ).to(device=batch_img.device)
            for i in range(len(batch_data_samples[0].metainfo["gt_depth"]))
        ]
        if depths is not None:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, gt_depth
            )
        return output

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_img = batch_inputs_dict["img"]
        feature_maps, _ = self.extract_feat(batch_img)
        # timestamp needs to be type double to avoid quantization errors
        timestamp = torch.tensor([bs.metainfo["timestamp"]
                                 for bs in batch_data_samples], dtype=torch.float64)
        model_outs = self.head(
            feature_maps,
            timestamp=timestamp,
            projection_mat=batch_inputs_dict["lidar2img"].to(torch.float32),
            # flip (H, W) to (W, H)
            image_wh=batch_inputs_dict["img_shape"][..., [1, 0]],
            batch_data_samples=batch_data_samples,
        )
        results = self.head.post_process(model_outs)
        output = self.add_pred_to_datasample(
            batch_data_samples, data_instances_3d=results
        )
        output = [op for op in output if not op.metainfo["padding"]]
        return output
