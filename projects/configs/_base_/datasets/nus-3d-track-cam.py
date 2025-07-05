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
input_shape = (704, 256)

strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(
        type='TrackLoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_forecasting=False),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    # dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type="Pack3DTrackInputs",
        keys=[
            "img",
            "lidar2img",
            "img_shape",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "instance_inds"
        ],
        meta_keys=["lidar2global", "timestamp", "intrinsics",
                   "lidar2img", "img_shape", "sample_idx", "scene_token",
                   "gt_depth"  # put depth in meta-keys since it doesn't stack easily into input or gt target structures
                   ],
    ),
]

# pass targets during val to compute losses/metrics
val_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(
        type='TrackLoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_forecasting=False),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="Pack3DTrackInputs",
        keys=[
            "img",
            "lidar2img",
            "img_shape",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "instance_inds"
        ],
        meta_keys=["lidar2global", "timestamp", "intrinsics",
                   "lidar2img", "img_shape", "sample_idx", "scene_token",
                   "gt_depth"  # put depth in meta-keys since it doesn't stack easily into input or gt target structures
                   ],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="Pack3DTrackInputs",
        keys=[
            "img",
            "lidar2img",
            "img_shape",
        ],
        meta_keys=["lidar2global", "timestamp", "intrinsics",
                   "lidar2img", "img_shape", "sample_idx", "scene_token"],
    ),
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
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
    # classes=class_names,
    modality=input_modality,
    data_prefix=data_prefix,
    # version="v1.0-trainval",
    metainfo=metainfo,
)
data_aug_conf = {
    "resize_lim": (0.40, 0.47),  # (640, 360) - (752, 423)
    "final_dim": input_shape,  # (704, 256), (W, H)
    "bot_pct_lim": (0.0, 0.0),  # unused?
    "rot_lim": (-5.4, 5.4),
    "W": 1600,
    "H": 900,
    "rand_img_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

# data_aug_conf = {
#     "resize_lim": (0.435, 0.435),
#     "final_dim": input_shape,
#     "bot_pct_lim": (0.0, 0.0),
#     "rot_lim": (0, 0),
#     "W": 1600,
#     "H": 900,
#     "rand_flip": False,
#     "rot3d_range": [0, 0],
# }

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
    collate_fn=dict(type='default_collate'),
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
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        **data_basic_config,
        ann_file=val_pkl_path,
        pipeline=val_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=False,
        verbose=True,
        filter_empty_gt=False,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesTrackingMetric',
    data_root=data_root,
    ann_file=data_root+val_pkl_path,
    metric='bbox',
    jsonfile_prefix='work_dirs/nuscenes_results/tracking',
    backend_args=backend_args)

test_evaluator = val_evaluator