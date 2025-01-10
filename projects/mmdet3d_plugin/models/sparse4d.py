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

from ..utils.misc import hash_tensor, hash_array

__all__ = ["Sparse4D"]


@MODELS.register_module()
class Sparse4D(MVXTwoStageDetector):
    def __init__(
        self,
        use_grid_mask: bool = True,
        use_deformable_func: bool = False,
        depth_branch: Optional[Dict]=None,
        freeze_pts: bool = True,
        **kwargs
    ):
        super(Sparse4D, self).__init__(**kwargs)
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
                True, True, offset=False, ratio=0.5, mode=1, prob=0.7
            )

        if freeze_pts:
            self.pts_backbone.eval()
            for param in self.pts_backbone.parameters():
                param.requires_grad = False

    def extract_img_feat(self, img: Optional[Tensor], return_depth: bool = False, batch_input_metas = None):
        if img is None:
            return None, None
        focal = torch.tensor([
            [intr[0, 0] for intr in bs["intrinsics"]]
            for bs in batch_input_metas], device=img.device)
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            # residual code from original Sparse4D
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

    def extract_feat(self, batch_inputs_dict: Dict, batch_input_metas: List[Dict]):
        # img feature extraction
        batch_img = batch_inputs_dict.get("img", None)
        feature_maps, depths = self.extract_img_feat(
            batch_img,
            return_depth=self.training,
            batch_input_metas=batch_input_metas,)

        # pts feature extraction
        pts_feats = self.extract_pts_feat(
            batch_inputs_dict.get('voxels', None),
            batch_input_metas=batch_input_metas,
        )

        if feature_maps is None:
            feature_maps = [None]
        if pts_feats is None:
            pts_feats = [None]
        
        # breakpoint() # check output of new_pts_feat against focalformer
        # new_img_feat, new_pts_feat = self.pts_fusion_layer(
            # feature_maps[0], pts_feats[0], batch_input_metas)
        return feature_maps, depths, new_pts_feat

    def loss(self, batch_inputs_dict: Dict,
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        # extract features
        new_img_feat, depths, new_pts_feat = self.extract_feat(
            batch_inputs_dict, batch_input_metas)
        # timestamp needs to be type double to avoid quantization errors
        timestamp = torch.tensor([bs.metainfo["timestamp"]
                                 for bs in batch_data_samples], dtype=torch.float64)
        model_outs = self.pts_bbox_head(
            new_img_feat,
            timestamp=timestamp,
            projection_mat=batch_inputs_dict["lidar2img"].to(torch.float32),
            # flip (H, W) to (W, H)
            image_wh=batch_inputs_dict["img_shape"][..., [1, 0]],
            batch_data_samples=batch_data_samples,
        )

        output = self.pts_bbox_head.loss(model_outs, batch_data_samples)

        gt_depth = [
            torch.from_numpy(
                np.stack([depth.metainfo["gt_depth"][i]
                         for depth in batch_data_samples])
            ).to(device=new_img_feat[0].device)
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
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        # extract features
        new_img_feat, depths, new_pts_feat = self.extract_feat(
            batch_inputs_dict, batch_input_metas)

        # timestamp needs to be type double to avoid quantization errors
        timestamp = torch.tensor([bs.metainfo["timestamp"]
                                 for bs in batch_data_samples], dtype=torch.float64)
        model_outs = self.pts_bbox_head(
            new_img_feat,
            timestamp=timestamp,
            projection_mat=batch_inputs_dict["lidar2img"].to(torch.float32),
            # flip (H, W) to (W, H)
            image_wh=batch_inputs_dict["img_shape"][..., [1, 0]],
            batch_data_samples=batch_data_samples,
        )
        results = self.pts_bbox_head.post_process(model_outs)
        output = self.add_pred_to_datasample(
            batch_data_samples, data_instances_3d=results
        )
        # output = [op for op in output if not op.metainfo["padding"]]
        return output
