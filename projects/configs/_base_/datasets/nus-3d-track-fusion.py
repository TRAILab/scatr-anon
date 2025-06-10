_base_ = [
    './nus-3d-track-lidar.py',
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_root = "data/nuscenes/"
class_names = {{_base_.class_names}}
point_cloud_range = {{_base_.point_cloud_range}}
points_loader = {{_base_.points_loader}}
image_size = (800, 448) # (width, height)

db_sampler = dict(
    type="TrackDBSampler",
    data_root=data_root,
    # info_path=data_root + 'nuscenes_track_dbinfos_train.pkl',
    info_path=data_root + 'fusion_ts_debug_track_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        # filter_by_difficulty=[], # no difficult in nuscenes
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2
    ),
    points_loader=points_loader,
    min_pixels=5,
)

train_pipeline = [
    points_loader,
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='TrackLoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_forecasting=False,
        with_mask=True
    ),
    # augmentations, kwargs in data_aug_conf
    dict(type='TrackSample', db_sampler=db_sampler, sample_2d=True),
    dict(type='SeqGlobalRotScaleTrans'),
    dict(type='SeqRandomFlip3D', sync_2d=False, flip_img=False),
    dict(type='PointShuffle'),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type='SeqImageAug3D',
        final_dim=image_size[::-1], # (height, width)
        resize_lim=[0.4, 0.6],
        bot_pct_lim=[0.0, 0.0],
        rand_flip=True,
        rot_lim=[-5.4, 5.4],
        is_train=False),
    # filter
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='TrackRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='TrackNameFilter', classes=class_names),
    # data packing
    dict(
        type="Pack3DTrackInputs",
        keys=[
            "points",
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "instance_inds",
        ],
        meta_keys=[
            "lidar2global", 
            "timestamp", 
            "sample_idx",
            "scene_token",
            "lidar_path"
            # check aug params
            "pcd_rotation_angle", 
            "pcd_trans", 
            "pcd_scale_factor",
            # fusion params
            "lidar2img",
            "cam2img",
            "img_aug_matrix"
        ],
    ),
]

# pass targets during val to compute losses/metrics
val_pipeline = [
    points_loader,
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # resize image
    dict(
        type='ImageAug3D',
        final_dim=image_size[::-1],
        resize_lim=[0.5, 0.5],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='TrackLoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_forecasting=False),
    dict(
        type="Pack3DTrackInputs",
        keys=[
            "points",
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "instance_inds",
        ],
        meta_keys=["lidar2global", "timestamp", "lidar2img",
                   "sample_idx", "scene_token", "lidar_path", "img_path", "num_pts_feats"],
    ),
]


data_aug_conf = dict(
    # lidar aug params
    rot_range_lidar=[-0.3925 * 2, 0.3925 * 2],
    scale_ratio_range_lidar=[0.9, 1.1],
    translation_std_lidar=[0.5, 0.5, 0.5],
    flip_ratio_bev_horizontal=0.5,
    flip_ratio_bev_vertical=0.5,
    use_track_sample_3d=True,
    # img resize params
    W=1600,
    H=900,
    final_dim=image_size,
    # ImageAug3D params
    resize_lim=[0.4, 0.6],
    rot_lim=[-5.4, 5.4],
    rand_img_flip=True,
)

data_aug_conf_test = dict(
    # lidar aug params
    rot_range_lidar=[0,0],
    scale_ratio_range_lidar=[1, 1],
    translation_std_lidar=[0, 0, 0],
    flip_ratio_bev_horizontal=0,
    flip_ratio_bev_vertical=0,
    use_track_sample_3d=False,
    # img resize params
    W=1600,
    H=900,
    final_dim=image_size,
    # ImageAug3D params
    resize_lim=[0.5, 0.5],
    rot_lim=[0, 0],
    rand_img_flip=False,
)


train_dataloader = dict(
    dataset=dict(
        data_aug_conf=data_aug_conf,
        pipeline=train_pipeline,
        modality=input_modality,
    )
)

val_dataloader = dict(
    dataset=dict(
        data_aug_conf=data_aug_conf_test,
        pipeline=val_pipeline,
        modality=input_modality
    )
)

test_dataloader = val_dataloader