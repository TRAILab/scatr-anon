_base_ = [
    './_base_/default_runtime.py',
    './_base_/datasets/nus-3d-track.py']

# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"

batch_size = 6
num_gpus = 8
total_batch_size = batch_size * num_gpus
input_shape = (704, 256)
num_epochs = 12
checkpoint_epoch_interval = 1
work_dir = f"work_dirs/sparse4dv3_temporal_r50_1x{num_gpus}_bs{batch_size}_{input_shape[1]}x{input_shape[0]}-{num_epochs}e_refactored-config"

load_from = None
# load_from = "ckpts/sparse4dv3_r50.pth"
resume_from = None

tracking_test = True
tracking_threshold = 0.2
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
# ================== model ========================

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
            num_cls={{_base_.num_classes}},
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
        decoder=dict(type="SparseBox3DDecoder",
                     score_threshold=tracking_threshold),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ================== data ========================
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
data_aug_conf = _base_.data_aug_conf
data_aug_conf.final_dim = input_shape

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        pipeline=train_pipeline,
        data_aug_conf=data_aug_conf
    )
)

val_dataloader = dict(
    batch_size=batch_size,
)
test_dataloader = dict(
    batch_size=batch_size,
)

# ================== training ========================
lr = 1e-5*total_batch_size  # 6e-4 for 8 gpus, bs=6
optim_wrapper = dict(
    type="OptimWrapper",
    # type="AmpOptimWrapper", # TODO does not work with Sparse4D upgrade yet
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.001),
    clip_grad=dict(max_norm=25, norm_type=2, error_if_nonfinite=True),
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
        by_epoch=True,
        eta_min=lr * 1e-3,
        convert_to_iter_based=True,
    )]

# runtime settings
train_cfg = dict(
    by_epoch=True,
    max_epochs=num_epochs,
    val_interval=checkpoint_epoch_interval)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(by_epoch=True,
                    interval=checkpoint_epoch_interval),
    logger=dict(interval=50)
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity="trailab",
            project="Sparse4Dv3-Lidar"),
    )
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

# set NCCL timeout to 3H to account for the validation computation taking longer than default 30 min on multiple GPUs
env_cfg = dict(
    dist_cfg=dict(timeout=10800),
)

# for debugging purposes, set deterministic=True
randomness = dict(seed=0, deterministic=True)
