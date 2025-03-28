from typing import Any, Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import (GlobalRotScaleTrans, ObjectNameFilter,
                                         ObjectRangeFilter, ObjectSample,
                                         RandomFlip3D)
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                LiDARInstance3DBoxes)
from PIL import Image


@TRANSFORMS.register_module()
class SeqGlobalRotScaleTrans(GlobalRotScaleTrans):
    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        aug_config = input_dict.get("aug_config")
        if aug_config is None:
            return input_dict

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict, aug_config['noise_rotation_lidar'])

        input_dict['pcd_scale_factor'] = aug_config['pcd_scale_factor']
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict, aug_config['trans_factor_lidar'])

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def _rot_bbox_points(self, input_dict: dict, noise_rotation) -> None:
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_bboxes_3d` is updated in the result dict.
        """

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            # rotate points with bboxes
            points, rot_mat_T = input_dict['gt_bboxes_3d'].rotate(
                noise_rotation, input_dict['points'])
            input_dict['points'] = points
        else:
            # if no bbox in input_dict, only rotate points
            rot_mat_T = input_dict['points'].rotate(noise_rotation)

        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation

    def _trans_bbox_points(self, input_dict: dict, trans_factor) -> None:
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'].translate(trans_factor)


@TRANSFORMS.register_module()
class SeqRandomFlip3D(RandomFlip3D):
    def __init__(self, flip_img:bool=False, **kwargs):
        super().__init__(**kwargs)
        self.flip_img = flip_img

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints.

        Add the override feature that if 'flip' is already in results, use it
        to do the augmentation.
        """
        # `flip_direction` works only when `flip` is True.
        # For example, in `MultiScaleFlipAug3D`, `flip_direction` is
        # 'horizontal' but `flip` is False.
        raise NotImplementedError(
            'flip_direction is not yet fully tested in SeqRandomFlip3D')
        aug_config = results["aug_config"]
        results['flip'] = aug_config['flip']
        results['flip_direction'] = aug_config['flip_direction']
        self._flip(results)

    def transform(self, input_dict: dict) -> dict:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
            'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
            into result dict.
        """
        aug_config = input_dict["aug_config"]

        # flip 2D image and its annotations
        if 'img' in input_dict and self.flip_img:
            super(RandomFlip3D, self).transform(input_dict)

        if self.sync_2d and 'img' in input_dict and self.flip_img:
            raise NotImplementedError(
                'sync_2d is not yet fully tested in SeqRandomFlip3D')
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                input_dict['pcd_horizontal_flip'] = aug_config['pcd_horizontal_flip']
            if 'pcd_vertical_flip' not in input_dict:
                input_dict['pcd_vertical_flip'] = aug_config['pcd_vertical_flip']

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict


@TRANSFORMS.register_module()
class TrackRangeFilter(ObjectRangeFilter):
    def transform(self, input_dict: dict) -> dict:
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds \
                keys are updated in the result dict. If 'gt_forecasting_locs' is in the \
                input_dict, 'gt_forecasting_locs', 'gt_forecasting_masks', 'gt_forecasting_types' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(
            input_dict["gt_bboxes_3d"], (LiDARInstance3DBoxes,
                                         DepthInstance3DBoxes)
        ):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict["gt_bboxes_3d"], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]
        else:
            raise TypeError(
                f"Invalid points instance type " f'{type(input_dict["gt_bboxes_3d"])}'
            )

        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        instance_inds = input_dict["instance_inds"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        instance_inds = instance_inds[mask.numpy().astype(bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        input_dict["instance_inds"] = instance_inds

        # hacks for forecasting
        if "gt_forecasting_locs" in input_dict.keys():
            input_dict["gt_forecasting_locs"] = input_dict["gt_forecasting_locs"][
                mask.numpy().astype(bool)
            ]
            input_dict["gt_forecasting_masks"] = input_dict["gt_forecasting_masks"][
                mask.numpy().astype(bool)
            ]
            input_dict["gt_forecasting_types"] = input_dict["gt_forecasting_types"][
                mask.numpy().astype(bool)
            ]

        return input_dict


@TRANSFORMS.register_module()
class TrackNameFilter(ObjectNameFilter):
    def transform(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds \
                keys are updated in the result dict. If 'gt_forecasting_locs' is in the \
                input_dict, 'gt_forecasting_locs', 'gt_forecasting_masks', 'gt_forecasting_types' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=bool)
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]

        # hacks for forecasting
        if "gt_forecasting_locs" in input_dict.keys():
            input_dict["gt_forecasting_locs"] = input_dict["gt_forecasting_locs"][
                gt_bboxes_mask
            ]
            input_dict["gt_forecasting_masks"] = input_dict["gt_forecasting_masks"][
                gt_bboxes_mask
            ]
            input_dict["gt_forecasting_types"] = input_dict["gt_forecasting_types"][
                gt_bboxes_mask
            ]

        return input_dict


@TRANSFORMS.register_module()
class TrackSample(ObjectSample):
    """
    TODO: support 2D sampling
    TODO: support ground plane sampling (only for KITTI dataset)
    """

    def __init__(self, sample_2d: bool = False, use_ground_plane: bool = False, **kwargs):
        assert not sample_2d, "2D sampling is not supported in TrackSample yet"
        assert not use_ground_plane, "Ground plane sampling is not supported in TrackSample yet"
        super().__init__(**kwargs)

    def transform(self, input_dict: dict) -> dict:
        sampled_track_list = input_dict["aug_config"].get("sampled_dict", None)

        if self.disabled or sampled_track_list is None:
            return input_dict
        # print([x['box3d_lidar'] for x in sampled_track_list])
        sampled_dict = self.db_sampler.sample_all(input_dict, sampled_track_list)
        if sampled_dict is None:
            return input_dict

        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_instance_inds = input_dict["instance_inds"]
        points = input_dict["points"]

        sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
        sampled_gt_labels = sampled_dict["gt_labels_3d"]
        sampled_gt_instance_inds = sampled_dict["instance_inds"]
        sampled_points = sampled_dict["points"]

        # add sampled annotations to the original gt annotations
        gt_labels_3d = np.concatenate(
            [gt_labels_3d, sampled_gt_labels], axis=0, dtype=np.int64
        )
        gt_bboxes_3d = gt_bboxes_3d.new_box(
            np.concatenate([gt_bboxes_3d.numpy(), sampled_gt_bboxes_3d])
        )
        gt_instance_inds = np.concatenate(
            [gt_instance_inds, sampled_gt_instance_inds], axis=0
        )

        if 'gt_forecasting_locs' in input_dict.keys():
            gt_forecasting_locs = input_dict["gt_forecasting_locs"]
            gt_forecasting_masks = input_dict["gt_forecasting_masks"]
            gt_forecasting_types = input_dict["gt_forecasting_types"]
            sampled_gt_forecasting_locs = sampled_dict["gt_forecasting_locs"]
            sampled_gt_forecasting_masks = sampled_dict["gt_forecasting_masks"]
            sampled_gt_forecasting_types = sampled_dict["gt_forecasting_types"]
            gt_forecasting_locs = np.concatenate(
                [gt_forecasting_locs, sampled_gt_forecasting_locs], axis=0
            )
            gt_forecasting_masks = np.concatenate(
                [gt_forecasting_masks, sampled_gt_forecasting_masks], axis=0
            )
            gt_forecasting_types = np.concatenate(
                [gt_forecasting_types, sampled_gt_forecasting_types], axis=0
            )

        # insert the sampled points into the points tensor
        points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
        points = points.cat([sampled_points, points])

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        input_dict["points"] = points
        input_dict["instance_inds"] = gt_instance_inds
        if 'gt_forecasting_locs' in input_dict.keys():
            input_dict["gt_forecasting_locs"] = gt_forecasting_locs
            input_dict["gt_forecasting_masks"] = gt_forecasting_masks
            input_dict["gt_forecasting_types"] = gt_forecasting_types

        return input_dict


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train,
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform)
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = torch.stack(transforms)
        return data

@TRANSFORMS.register_module()
class SeqImageAug3D(ImageAug3D):
    def sample_augmentation(self, results):
        aug_config = results.get("aug_config")
        if aug_config:
            resize = aug_config["resize"]
            resize_dims = aug_config["resize_dims"]
            crop = aug_config["crop"]
            flip = aug_config["flip"]
            rotate = aug_config["rotate"]
        else:
            resize, resize_dims, crop, flip, rotate = super().sample_augmentation(results)
        return resize, resize_dims, crop, flip, rotate