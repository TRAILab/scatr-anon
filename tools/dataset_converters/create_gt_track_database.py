# Based on tools/dataset_converters/create_gt_database.py
import pickle
from os import path as osp

import mmcv
import mmengine
import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops
from mmdet.evaluation import bbox_overlaps
from mmengine import track_iter_progress

from projects.mmdet3d_plugin.datasets import NuScenes3DDetTrackDataset

from .create_gt_database import crop_image_patch, crop_image_patch_v2


def create_groundtruth_track_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    with_mask:bool=False,
):
    """Given the raw data, generate the ground truth database.
    The database contains track trajectories all together

    Currently only support NuScenes dataset.
    # TODO support other datasets

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
    """
    print(f'Create GT Database of {dataset_class_name}')
    assert dataset_class_name in [
        'NuScenesTrackingDataset'
    ], 'Only support NuScenesTrackingDataset for now'
    dataset_cfg = dict(
        type=dataset_class_name,
        data_root=data_path,
        ann_file=info_path,
        use_valid_flag=True,
        modality=dict(
            use_lidar=True,
            use_camera=True
        ),
        data_prefix=dict(
            pts='samples/LIDAR_TOP', 
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            sweeps='sweeps/LIDAR_TOP'),
        pipeline=[
            dict(type='LoadPointsFromFile',
                 coord_type='LIDAR', load_dim=5, use_dim=5),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                pad_empty_sweeps=True,
                remove_close=True,
            ),
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='TrackLoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_forecasting=False,
                with_mask=True,
            ),
        ],
    )

    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(
            data_path, f'{info_prefix}_track_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(
            data_path, f'{info_prefix}_track_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    group_counter = 0
    # iterate through each sample in the dataset
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        if len(data_info['ann_info']['gt_labels_3d']) == 0:
            continue
        example = dataset.pipeline(data_info)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        gt_labels = [dataset.metainfo['classes'][i]
                     for i in annos['gt_labels_3d']]
        instance_inds = annos['instance_inds']
        img = np.stack(example['img'], axis=0)
        gt_masks = example.get('gt_masks', None)
        gt_mask_pos = example.get('gt_mask_pos', None)
        num_cams = len(gt_masks)
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for obj_idx in range(num_obj): # iterate through each object in the sample
            # TODO directly iterate over indices with enumerate and zip rather than directly indexing
            if used_classes is not None and gt_labels[obj_idx] not in used_classes:
                # skip unused class
                continue
            filename = f'{image_idx}_{gt_labels[obj_idx]}_{obj_idx}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(
                f'{info_prefix}_track_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, obj_idx]]
            # center the points about the object
            gt_points[:, :3] -= gt_boxes_3d[obj_idx, :3]

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            db_info = {
                'name': gt_labels[obj_idx],
                'path': rel_filepath,
                'image_idx': image_idx,
                'gt_idx': obj_idx,
                'box3d_lidar': gt_boxes_3d[obj_idx],
                'num_points_in_gt': gt_points.shape[0],
                'difficulty': difficulty[obj_idx],
                'valid': True,
                'instance_ind': instance_inds[obj_idx],
            }

            local_group_id = group_ids[obj_idx]
            if local_group_id not in group_dict:
                group_dict[local_group_id] = group_counter
                group_counter += 1
            db_info['group_id'] = group_dict[local_group_id]
            if 'score' in annos:
                db_info['score'] = annos['score'][obj_idx]
            if gt_labels[obj_idx] not in all_db_infos:
                all_db_infos[gt_labels[obj_idx]] = {}
            if instance_inds[obj_idx] not in all_db_infos[gt_labels[obj_idx]]:
                all_db_infos[gt_labels[obj_idx]][instance_inds[obj_idx]] = []

            if not with_mask:
                all_db_infos[gt_labels[obj_idx]][instance_inds[obj_idx]].append(db_info)
                continue

            # iterate through each camera
            img_paths = [None] * num_cams
            gt_boxes = [None] * num_cams
            for cam_idx, (gt_mask, mask_pos) in enumerate(zip(gt_masks[obj_idx], gt_mask_pos[obj_idx])):
                if gt_mask is None or mask_pos is None:
                    continue
                img_patch_path = abs_filepath + f'cam_{cam_idx}.png'
                rel_path = osp.join(
                    f'{info_prefix}_track_gt_database', 
                    filename + f'cam_{cam_idx}.png')
                x1, y1, w, h = mask_pos
                # xyxy format
                gt_boxes[cam_idx] = np.array([x1, y1, x1 + w, y1 + h])
                masked_patch = img[cam_idx, y1:y1+h, x1:x1+w].copy() * \
                    gt_mask
                mmcv.imwrite(masked_patch, img_patch_path)
                img_paths[cam_idx] = rel_path
            db_info.update({
                'box2d_camera': gt_boxes,
                'img_path': img_paths,
            })
            all_db_infos[gt_labels[obj_idx]][instance_inds[obj_idx]].append(db_info)

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


'''Cannot parallelize the creation of gt track database easily, since the operation is not 
independent for each sample. The creation of the track DB is sequential.

The parallelization would have to be at the level of the scene.
'''
