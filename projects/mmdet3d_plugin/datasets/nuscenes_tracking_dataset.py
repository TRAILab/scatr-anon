# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
from typing import List, Union

import numpy as np
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS
from mmengine.logging import print_log

from projects.mmdet3d_plugin.datasets.pipelines import TrackSample


@DATASETS.register_module()
class NuScenesTrackingDataset(NuScenesDataset):
    def __init__(self,
                 *args,
                 forecasting: bool = False,
                 seq_split_num: int = 2,
                 data_aug_conf: dict = {},
                 **kwargs,
                 ):
        self.forecasting = forecasting
        self.scene_tokens = []
        self.scene_tokens_2_instance_inds = {}
        self.scene_tokens_2_num_classes = {}
        self.indices_per_scene = {}
        self.seq_split_num = seq_split_num
        self.cls_distr = []
        assert self.seq_split_num >= 1
        self.data_aug_conf = data_aug_conf
        super().__init__(*args, **kwargs)
        # resize params
        H, W = self.data_aug_conf.pop("H"), self.data_aug_conf.pop("W")
        fH, fW = self.data_aug_conf.pop("final_dim")
        self.resize = max(fH/H, fW/W)
        self.resize_dims = (int(W * self.resize), int(H * self.resize))
        newW, newH = self.resize_dims
        crop_h = (newH- fH)
        crop_w = int(max(0, newW - fW) / 2)
        self.crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        # check resize_dims matches with final_dim
        if self.resize_dims != (fW, fH):
            print_log(
                f"resize_dims {self.resize_dims} does not match final_dim {fW, fH}"
            )

        if self.test_mode and self.data_aug_conf != {}:
            print_log(
                "Data augmentation is not allowed in test mode. "
                "The data augmentation configuration will be ignored.",
                "current",
                level=30,
            )
            self.data_aug_conf = {}

    def get_augmentation(self, clip_inds: List[int]):
        """
        Imported from Sparse4Dv3
        """
        aug_config = {}
        # Resize
        aug_config["resize"] = self.resize
        aug_config["crop"] = self.crop
        # BEVFusionRandomFlip3D
        aug_config["flip_horizontal_3d"] = self.data_aug_conf.get(
            "random_flip_3d_horiz", False) and np.random.choice([0, 1])
        aug_config["flip_vertical_3d"] = self.data_aug_conf.get(
            "random_flip_3d_vert", False) and np.random.choice([0, 1])
        # BEVFusionGlobalRotScaleTrans
        aug_config["rotation_3d"] = np.random.uniform(
            *self.data_aug_conf.get("rotation_3d_range", [0, 0]))
        aug_config["translation_3d"] = np.random.normal(
            *self.data_aug_conf.get("translation_3d_range", [0, 0]), size=3).T
        aug_config["scale_3d"] = np.random.uniform(
            *self.data_aug_conf.get("scale_3d_range", [1, 1]))
        aug_config_list = [copy.deepcopy(aug_config)
                           for _ in range(len(clip_inds))]
        # TrackDBSampler
        if self.data_aug_conf.get("use_track_sample_3d", False):
            assert any([isinstance(tf, TrackSample) for tf in self.pipeline.transforms]), \
                "track sample 3d is set to true in the data_aug_conf of NuScenesTrackingDataset but the pipeline does not contain TrackSampler3D"
            track_db_sampler = (tf.db_sampler for tf in self.pipeline.transforms if isinstance(
                tf, TrackSample)).__next__()
            scene_token = self.get_scene_token(clip_inds[0])
            track_sample_dict_list = track_db_sampler.get_samples(
                [self.cls_distr[i] for i in clip_inds], scene_token)
            for track_sample_dict_i, aug_conf in zip(track_sample_dict_list, aug_config_list):
                aug_conf["sampled_dict"] = track_sample_dict_i
        return aug_config_list

    def prepare_data(self, index) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        if isinstance(index, dict):
            aug_config = copy.deepcopy(index["aug_config"])
            new_scene = index["new_scene"]
            padding = index["padding"]
            index = index["index"]
        else:
            assert isinstance(
                index, int), f"index must be int, but got {type(index)}"
            print_log(
                "The input index is not a dict. It's probably that the data sampler is not TrackSampler3D. The augmentation may be different for each sample",
            )
            aug_config = self.get_augmentation([index])[0]
            new_scene = False
            padding = False
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                print("empty gt before pipeline at index: ", index)
                return None

        input_dict["aug_config"] = aug_config
        example = self.pipeline(input_dict)

        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['data_samples'].gt_instances_3d.labels_3d) == 0:
                print("empty gt after pipeline at index: ", index)
                return None

        if self.show_ins_var:
            if 'ann_info' in ori_input_dict:
                self._show_ins_var(
                    ori_input_dict['ann_info']['gt_labels_3d'],
                    example['data_samples'].gt_instances_3d.labels_3d)
            else:
                print_log(
                    "'ann_info' is not in the input dict. It's probably that "
                    'the data is not in training mode',
                    'current',
                    level=30)
        example['data_samples'].set_metainfo(
            {'new_scene': new_scene, "padding": padding})
        return example

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """
        Called once at initialization
        """
        data_info = super().parse_data_info(info)
        scene_token = data_info['scene_token']
        self.scene_tokens.append(scene_token)
        self.indices_per_scene[scene_token] = self.indices_per_scene.get(
            scene_token, []) + [len(self.scene_tokens) - 1]

        # use bincount of classes for track sampling
        if self.test_mode:
            ann_info = data_info['eval_ann_info']
        else:
            ann_info = data_info['ann_info']
        self.cls_distr.append(np.bincount(
            [x for x in ann_info['gt_labels_3d'] if x >= 0], minlength=len(self.metainfo['classes'])))

        # ego movement represented by lidar2global
        l2e = np.array(data_info['lidar_points']['lidar2ego'])
        e2g = np.array(data_info['ego2global'])
        l2g = e2g @ l2e

        # points @ R.T + T
        data_info.update(lidar2global=l2g.astype(np.float32))

        if self.modality['use_camera']:
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['images'].items():
                # obtain lidar to image transformation matrix
                lidar2cam_rt = np.array(cam_info['lidar2cam']).T
                intrinsic = np.array(cam_info['cam2img'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                intrinsics.append(viewpad)
                # the transpose of extrinsic matrix
                extrinsics.append(lidar2cam_rt)

            data_info.update(
                dict(
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                ))
        return data_info

    def get_num_scenes(self):
        return len(set(self.scene_tokens))

    def get_scene_token(self, index):
        return self.scene_tokens[index]

    def get_scene_token_indices(self, scene_token):
        return self.indices_per_scene[scene_token]

    def get_all_scene_tokens(self):
        return sorted(list(set(self.scene_tokens)))

    def get_len_per_video(self, idx):
        return len(self.get_scene_token_indices(self.get_scene_token(idx)))

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        raise Exception("This function should not be called")
