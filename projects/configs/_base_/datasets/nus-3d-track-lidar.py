class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

num_classes = len(class_names)
dataset_type = "NuScenesTrackingDataset"
data_root = "data/nuscenes/"
anno_root = ""
train_pkl_path = anno_root + "nuscenes_sparse4d_mmlabv2_11-06_infos_train.pkl"
val_pkl_path = anno_root + "nuscenes_sparse4d_mmlabv2_11-06_infos_val.pkl"
# train_pkl_path = anno_root + "nuscenes_sparse4d_mmlabv2_11-18_mini_infos_val.pkl"
# val_pkl_path = anno_root + "nuscenes_sparse4d_mmlabv2_11-18_mini_infos_val.pkl"
# train_pkl_path = anno_root + "nusc-mini-np1-mmv2_infos_val.pkl"
# val_pkl_path = anno_root + "nusc-mini-np1-mmv2_infos_val.pkl"
backend_args = None

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

points_loader = dict(
    type="LoadPointsFromFile",
    coord_type="LIDAR",
    load_dim=5,
    use_dim=5,
    backend_args=backend_args,
)

db_sampler = dict(
    type="TrackDBSampler",
    data_root=data_root,
    info_path=data_root + 'nuscenes_track_dbinfos_train.pkl',
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
    sample_2d=False
)


train_pipeline = [
    points_loader,
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='TrackLoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_forecasting=False),
    # augmentations, kwargs in data_aug_conf
    dict(type='TrackSample', db_sampler=db_sampler),
    dict(type='SeqGlobalRotScaleTrans'),
    dict(type='SeqRandomFlip3D', sync_2d=False),
    dict(type='PointShuffle'),
    # filter
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='TrackRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='TrackNameFilter', classes=class_names),
    # data packing
    dict(
        type="Pack3DTrackInputs",
        keys=[
            "points",
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
            "gt_bboxes_3d",
            "gt_labels_3d",
            "instance_inds",
        ],
        meta_keys=["lidar2global", "timestamp", "lidar2img",
                   "sample_idx", "scene_token", "lidar_path", "img_path", "num_pts_feats"],
    ),
]

test_pipeline = [
    points_loader,
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="Pack3DTrackInputs",
        keys=["points",],
        meta_keys=["lidar2global", "timestamp", "lidar2img",
                   "sample_idx", "scene_token", "lidar_path", "img_path", "num_pts_feats"],
    ),
]

# aug params from FocalFormer3D
data_aug_conf = dict(
    # lidar aug params
    rot_range_lidar=[-0.3925 * 2, 0.3925 * 2],
    scale_ratio_range_lidar=[0.9, 1.1],
    translation_std_lidar=[0.5, 0.5, 0.5],
    flip_ratio_bev_horizontal=0.5,
    flip_ratio_bev_vertical=0.5,
    use_track_sample_3d=True,
    W=1600,
    H=900,
    final_dim= (704, 256),
    bot_pct_lim= (0.0, 0.0)
)

data_aug_conf_eval = dict(
    W=1600,
    H=900,
    final_dim= (704, 256),
    bot_pct_lim= (0.0, 0.0)
)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')

metainfo = dict(classes=class_names, version='v1.0-trainval')

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    modality=input_modality,
    data_prefix=data_prefix,
    metainfo=metainfo,
)

batch_size = 2

train_dataloader = dict(
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler'),
    batch_sampler=dict(
        type='TrackSampler3D', 
        shuffle=True,
        max_clip_len=10,
        # clip_len=10,
        # num_splits=2,
        seq_flip_prob=0.1, 
        use_CBGS=True
    ),
    # collate_fn=dict(type='default_collate'),
    dataset=dict(
        **data_basic_config,
        ann_file=train_pkl_path,
        pipeline=train_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=False,
        # we should still be able to train on empty GT, and breaks stream training
        filter_empty_gt=False,
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler'),
    batch_sampler=dict(type='TrackSampler3D', shuffle=False, clip_len=-1),
    # collate_fn=dict(type='default_collate'),
    dataset=dict(
        **data_basic_config,
        ann_file=val_pkl_path,
        pipeline=val_pipeline,
        data_aug_conf=data_aug_conf_eval,
        test_mode=False,
        verbose=True,
        filter_empty_gt=False,
    )
)

test_dataloader = val_dataloader
# test_dataloader['dataset']['pipeline'] = test_pipeline
# test_dataloader['dataset']['test_mode'] = True

val_evaluator = dict(
    type='NuScenesTrackingMetric',
    data_root=data_root,
    ann_file=data_root+val_pkl_path,
    metric='bbox',
    jsonfile_prefix='work_dirs/nuscenes_results/tracking',
    backend_args=backend_args)

test_evaluator = val_evaluator
