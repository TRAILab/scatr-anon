_base_ = [
    './_base_/default_runtime.py',
    './_base_/datasets/nus-3d-track-lidar.py',
    './_base_/models/sparse4dv3_temporal_SECOND.py',]

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
val_epoch_interval = 1

short_name = "lidar-supervise_qc"
work_dir = f"work_dirs/sparse4dv3-temporal_lidar_1x{num_gpus}_bs{batch_size}_{input_shape[1]}x{input_shape[0]}-{num_epochs}e_{short_name}"

load_from = 'ckpts/focalformer3d_converted/DeformFormer3D_L_iterimg_ep20_mAP655_NDS707.pth'
resume_from = None

tracking_test = True
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

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# ================== model ========================

strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
        )
    ),
    pts_bbox_head=dict(
        point_cloud_range=point_cloud_range,
        refine_layer=dict(
            num_cls={{_base_.num_classes}}, # from dataset
        ),
        sampler=dict(
            supervise_qc=True,
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
        loss_reg=dict(
            cls_allow_reverse=[class_names.index("barrier")],
        )
    )
)

# ================== data ========================
train_dataloader = dict(
    batch_size=batch_size,
)

val_dataloader = dict(
    batch_size=batch_size,
)
test_dataloader = dict(
    batch_size=batch_size,
)

# ================== training ========================
lr = 1.0e-5*total_batch_size  # 6e-4 for 8 gpus, bs=6
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
    val_interval=val_epoch_interval)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(by_epoch=True,
                    interval=checkpoint_epoch_interval),
    logger=dict(interval=50)
)

disable_ts_ratio = 0.75
custom_hooks = [
    dict(
        type="DisableTrackSampleHook",
        disable_after_epoch=int(num_epochs * disable_ts_ratio),
    )
]

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type='WandbVisBackend',
        save_dir=work_dir,
        init_kwargs=dict(
            entity="trailab",
            project="Sparse4Dv3-Lidar",
            name=short_name,
        )
    )
]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)


custom_imports = dict(
    imports=["projects.mmdet3d_plugin"], allow_failed_imports=False)

# set NCCL timeout to 3H to account for the validation computation taking longer than default 30 min on multiple GPUs
env_cfg = dict(
    dist_cfg=dict(timeout=10800),
)

# for debugging purposes, set deterministic=True
randomness = dict(seed=0, deterministic=True)
