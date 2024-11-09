"""
mAP: 0.4647
mATE: 0.5403
mASE: 0.2623
mAOE: 0.4590
mAVE: 0.2198
mAAE: 0.2059
NDS: 0.5636
Eval time: 176.9s

Per-class results:
Object Class    AP  ATE ASE AOE AVE AAE
car 0.668   0.357   0.142   0.054   0.184   0.195
truck   0.394   0.528   0.187   0.052   0.163   0.210
bus 0.451   0.681   0.196   0.070   0.383   0.243
trailer 0.185   0.971   0.247   0.634   0.175   0.202
construction_vehicle    0.122   0.879   0.496   1.200   0.136   0.406
pedestrian  0.559   0.517   0.287   0.513   0.282   0.151
motorcycle  0.497   0.462   0.238   0.536   0.293   0.236
bicycle 0.426   0.441   0.257   0.951   0.142   0.004
traffic_cone    0.697   0.275   0.299   nan nan nan
barrier 0.648   0.292   0.275   0.122   nan nan
"""

"""
Per-class results:
            AMOTA   AMOTP   RECALL  MOTAR   GT      MOTA    MOTP    MT  ML  FAF     TP      FP  FN  IDS FRAG TID    LGD
bicycle     0.444   1.169   0.533   0.733   1993    0.389   0.566   53  57  19.3    1059    283 931 3   8   1.60    1.75
bus         0.559   1.175   0.626   0.824   2112    0.515   0.751   42  35  14.8    1321    233 790 1   20  1.13    1.95
car         0.678   0.755   0.733   0.819   58317   0.599   0.470   2053    1073    134.2   42626   7706    15565   126 295 0.76    1.03
motorcy     0.522   1.060   0.609   0.823   1977    0.497   0.564   50  38  15.7    1194    211 773 10  17  1.97    2.17
pedestr     0.548   1.059   0.652   0.791   25423   0.506   0.678   677 467 77.6    16274   3404    8854    295 225 1.33    1.85
trailer     0.136   1.603   0.383   0.403   2425    0.154   0.981   30  79  52.6    926 553 1496    3   13  1.49    2.64
truck       0.454   1.132   0.577   0.691   9650    0.399   0.594   210 214 45.7    5569    1723    4078    3   50  1.35    1.85

Aggregated results:
AMOTA   0.477
AMOTP   1.136
RECALL  0.588
MOTAR   0.726
GT  14556
MOTA    0.437
MOTP    0.658
MT  3115
ML  1963
FAF 51.4
TP  68969
FP  14113
FN  32487
IDS 441
FRAG    628
TID 1.37
LGD 1.89
"""

_base_ = ['./_base_/default_runtime.py']

# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "work_dirs/sparse4dv3_temporal_r50_1x8_bs6_256x704"

total_batch_size = 6
num_gpus = 1
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)

tracking_test = True
tracking_threshold = 0.2

# ================== model ========================
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
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True

model = dict(
    type="Sparse4D",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=(
            dict(type="Pretrained", checkpoint="ckpts/resnet50-19c8e357.pth"),
        )
    ),
    img_neck=dict(
        type="mmdet.FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="_nuscenes_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        )
        if temporal
        else None,
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="mmdet.L1Loss", loss_weight=0.25),
            loss_centerness=dict(
                type="mmdet.CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="mmdet.GaussianFocalLoss"),
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ================== data ========================
dataset_type = "NuScenesTrackingDataset"
data_root = "data/nuscenes/"
anno_root = ""
train_pkl_path = anno_root + "nuscenes_sparse4d_mmlabv2_11-06_infos_train.pkl"
val_pkl_path = anno_root + "nuscenes_sparse4d_mmlabv2_11-06_infos_val.pkl"
backend_args = None

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
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type='TrackLoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_forecasting=False),
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
                   "gt_depth" # put depth in meta-keys since it doesn't stack easily into input or gt target structures
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
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler'),
    batch_sampler=dict(type='TrackSampler3D', shuffle=True, clip_len=20),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        **data_basic_config,
        ann_file=train_pkl_path,
        pipeline=train_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=False,
        filter_empty_gt=False, # we should still be able to train on empty GT, and breaks stream training
        # tracking=tracking_test,
        # tracking_threshold=tracking_threshold
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
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        # tracking=tracking_test,
        # tracking_threshold=tracking_threshold
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

# ================== training ========================
lr = 7.5e-5*num_gpus
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.001),
    clip_grad=dict(max_norm=25, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type="CosineAnnealingLR",
        begin=500,
        T_max=num_iters_per_epoch * num_epochs,
        end=num_iters_per_epoch * num_epochs,
        by_epoch=False,
        eta_min=lr * 1e-3
    )]

# runtime settings
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=num_iters_per_epoch * num_epochs,
    val_interval=num_iters_per_epoch * checkpoint_epoch_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(by_epoch=False,
                    interval=num_iters_per_epoch * checkpoint_epoch_interval),
    logger=dict(interval=50)
    )

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# ================== eval ========================
# vis_pipeline = [
#     dict(type="LoadMultiViewImageFromFiles", to_float32=True),
#     dict(
#         type="Pack3DDetInputs",
#         keys=["img"],
#         meta_keys=["timestamp", "lidar2img"],
#     ),
# ]
# evaluation = dict(
#     interval=num_iters_per_epoch * checkpoint_epoch_interval,
#     pipeline=vis_pipeline,
#     # out_dir="./vis",  # for visualization
# )

custom_imports = dict(
    imports=["projects.mmdet3d_plugin"], allow_failed_imports=False)
