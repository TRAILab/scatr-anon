from inspect import signature
from typing import Dict, List, Optional

import numpy as np
import torch
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import BaseDataElement
from torch import Tensor, nn

from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

from projects.mmdet3d_plugin.utils.misc import hash_array  # debug tools
from projects.mmdet3d_plugin.utils.misc import hash_tensor

__all__ = ["SCATr"]


@MODELS.register_module()
class SCATr(MVXTwoStageDetector):
    def __init__(
        self,
        use_grid_mask: bool = True,
        use_deformable_func: bool = False,
        depth_branch: Optional[Dict] = None,
        freeze_pts: bool = True,
        freeze_img: bool = True,
        freeze_camlss: bool = True,
        freeze_fusion: bool = False,
        **kwargs
    ):
        super(SCATr, self).__init__(**kwargs)
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

        if freeze_pts and self.with_pts_backbone:
            self.pts_backbone.eval()
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
            self.pts_middle_encoder.eval()
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
            self.pts_neck.eval()
            for param in self.pts_neck.parameters():
                param.requires_grad = False

            # Fix bn, from focalformer3d
            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
            self.pts_voxel_encoder.apply(fix_bn)
            self.pts_middle_encoder.apply(fix_bn)
            self.pts_backbone.apply(fix_bn)
            self.pts_neck.apply(fix_bn)

        if freeze_img and self.with_img_backbone:
            self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
            if self.with_img_neck:
                self.img_neck.eval()
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.with_pts_fusion_layer and freeze_camlss and hasattr(self.pts_fusion_layer, 'cam_lss'):
                self.pts_fusion_layer.cam_lss.eval()
                for param in self.pts_fusion_layer.cam_lss.parameters():
                    param.requires_grad = False

        if freeze_fusion and self.with_pts_fusion_layer:
            self.pts_fusion_layer.eval()
            for param in self.pts_fusion_layer.parameters():
                param.requires_grad = False

    def extract_img_feat(self, batch_inputs_dict, return_depth: bool = False, batch_input_metas=None):
        """
        NOTE: when using dataset preprocessor, the input key is 'imgs'.
        When not using dataset preprocessor (cam only), the input key is 'img'.
        """
        if self.with_pts_voxel_encoder: # use simple feat extraction for LC fusion
            img = batch_inputs_dict.get('imgs', None)
            img_feat = super().extract_img_feat(img, batch_input_metas)
            return img_feat, None
        img = batch_inputs_dict.get('img', None)
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
        # output of preprocessor is imgs
        feature_maps, depths = self.extract_img_feat(
            batch_inputs_dict,
            return_depth=self.training,
            batch_input_metas=batch_input_metas,)

        # pts feature extraction
        if self.with_pts_voxel_encoder:
            pts_feats = self.extract_pts_feat(
                batch_inputs_dict.get('voxels', None),
                batch_input_metas=batch_input_metas,
            )
        else:
            pts_feats = None
        # hash_tensor(pts_feats[0]) = 47c0c66f09f334cc63bff52660e5dcc05236722a with no spconv
        # 0ea90631caf8286115d6cfb827a21017b09398f6 with SPCONV
        if feature_maps is None:
            feature_maps = [None]
        if pts_feats is None:
            pts_feats = [None]

        # TODO check output of new_pts_feat against focalformer, need the same torch/cuda version for reproducibility
        # breakpoint()
        if self.with_pts_fusion_layer:
            new_img_feat, new_pts_feat = self.pts_fusion_layer(
                feature_maps[0], pts_feats[0], batch_input_metas)
            # new_img_feat is not actually used in focalformer head
            return feature_maps, depths, new_pts_feat
        else:  # just return the normal features
            return feature_maps, depths, pts_feats

    def loss(self, batch_inputs_dict: Dict,
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        # print([batch_data_samples[0].metainfo[x] for x in ["scene_token", "pcd_rotation_angle", "pcd_trans", "pcd_scale_factor"]])
        # print([batch_data_samples[1].metainfo[x] for x in ["scene_token", "pcd_rotation_angle", "pcd_trans", "pcd_scale_factor"]])
        # breakpoint()
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        # extract features
        new_img_feat, depths, new_pts_feat = self.extract_feat(
            batch_inputs_dict, batch_input_metas)
        # timestamp needs to be type double to avoid quantization errors
        timestamp = torch.tensor([bs.metainfo["timestamp"]
                                 for bs in batch_data_samples], dtype=torch.float64)

        # handle camera-specific data
        if 'lidar2img' in batch_inputs_dict:
            lidar2img = batch_inputs_dict['lidar2img'].to(torch.float32)
        else:
            lidar2img = None
        if 'img_shape' in batch_inputs_dict:
            # flip (H, W) to (W, H)
            image_wh = batch_inputs_dict['img_shape'][..., [1, 0]]
        else:
            image_wh = None

        model_outs = self.pts_bbox_head(
            new_pts_feat,
            new_img_feat,
            timestamp=timestamp,
            projection_mat=lidar2img,
            image_wh=image_wh,
            batch_data_samples=batch_data_samples,
        )

        output = self.pts_bbox_head.loss(model_outs, batch_data_samples)

        if depths is not None:
            gt_depth = [
                torch.from_numpy(
                    np.stack([depth.metainfo["gt_depth"][i]
                              for depth in batch_data_samples])
                ).to(device=new_img_feat[0].device)
                for i in range(len(batch_data_samples[0].metainfo["gt_depth"]))
            ]
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
        # handle camera-specific data
        if 'lidar2img' in batch_inputs_dict:
            lidar2img = batch_inputs_dict['lidar2img'].to(torch.float32)
        else:
            lidar2img = None
        if 'img_shape' in batch_inputs_dict:
            # flip (H, W) to (W, H)
            image_wh = batch_inputs_dict['img_shape'][..., [1, 0]]
        else:
            image_wh = None

        model_outs = self.pts_bbox_head(
            new_pts_feat,
            new_img_feat,
            timestamp=timestamp,
            projection_mat=lidar2img,
            image_wh=image_wh,
            batch_data_samples=batch_data_samples,
        )

        # compute val losses
        loss_dict = self.pts_bbox_head.loss(model_outs, batch_data_samples)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        if depths is not None:
            gt_depth = [
                torch.from_numpy(
                    np.stack([depth.metainfo["gt_depth"][i]
                              for depth in batch_data_samples])
                ).to(device=new_img_feat[0].device)
                for i in range(len(batch_data_samples[0].metainfo["gt_depth"]))
            ]
            loss_dict["val_loss_dense_depth"] = self.depth_branch.loss(
                depths, gt_depth
            )

        results = self.pts_bbox_head.post_process(model_outs)
        output = self.add_pred_to_datasample(
            batch_data_samples, data_instances_3d=results
        )

        # add loss dict to output
        output.append(BaseDataElement(loss=loss_dict))
        return output

    @property
    def with_pts_fusion_layer(self):
        """bool: Whether the detector has a fusion layer.
        Original MVXTwoStageDetector has a typo, calls self.fusion_layer instead of self.pts_fusion_layer"""
        return hasattr(self, 'pts_fusion_layer') and self.pts_fusion_layer is not None

    @property
    def with_pts_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'pts_voxel_encoder') and self.pts_voxel_encoder is not None
