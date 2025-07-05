_base_ = [
    './_base_/default_runtime.py',
    './_base_/datasets/nus-3d-track-fusion.py',
    './_base_/models/sparse4dv3-LC.py',]

# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"

batch_size = 2
num_gpus = 8
total_batch_size = batch_size * num_gpus
num_epochs = 10
checkpoint_epoch_interval = 1
val_epoch_interval = 1
image_size = (800, 448) # (width, height)

short_name = "LC-sp4dL-pretrained-mm_cutpaste"
work_dir = f"work_dirs/sparse4dv3-LC_nusc-{num_gpus}_bs{batch_size}_{num_epochs}e_{short_name}"

# load_from = 'ckpts/focalformer3d_converted/FocalFormer3D_LC_ep6_mAP705_NDS731.pth'
load_from = 'work_dirs/sparse4dv3-temporal_lidar_1x4_bs6-10e_lidar-4g-emb256_new-bn/epoch_10.pth'
# resume = True

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
    pts_fusion_layer=dict(
        pc_range=point_cloud_range,
        img_scale=(image_size[1], image_size[0]),
    ),
    pts_bbox_head=dict(
        point_cloud_range=point_cloud_range,
        instance_bank=dict(
            class_names=class_names,
            num_anchor=900,
            anchor="_nuscenes_kmeans900.npy",
            num_temp_instances=600,
            dataset_name={{_base_.dataset_type}},
            feat_pool=True,
            num_learned_groups=1,
            num_learned_temp_groups=1,
            group_selection=["topk"],
        ),
        anchor_encoder=dict(
            output_fc=True,
            output_dim={{_base_.embed_dims}},
        ),
        refine_layer=dict(
            num_cls={{_base_.num_classes}}, # from dataset
        ),
        sampler=dict(
            feat_pool=True,
            second_chance_tq=True,
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

val_evaluator = dict(
    jsonfile_prefix=f"{work_dir}/nuscenes_val",
)

test_evaluator = dict(
    jsonfile_prefix=f"{work_dir}/nuscenes_test",
)

# ================== training ========================
lr = 1.0e-5*total_batch_size  # 6e-4 for 8 gpus, bs=6
optim_wrapper = dict(
    type="OptimWrapper",
    # type="AmpOptimWrapper", # TODO does not work with Sparse4D upgrade yet
    # loss_scale=1.0,
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.001),
    clip_grad=dict(max_norm=25, norm_type=2, error_if_nonfinite=True),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)

# param_scheduler = [
#     dict(
#         type="LinearLR",
#         start_factor=1.0/3,
#         by_epoch=False,
#         begin=0,
#         end=500
#     ),
#     dict(
#         type="CosineAnnealingLR",
#         by_epoch=True,
#         eta_min=lr * 1e-3,
#         convert_to_iter_based=True,
#     )]
param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=lr,
        total_steps=num_epochs,
        pct_start=0.4,
        div_factor=25.0,
        final_div_factor=1e4,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=(0.4 * num_epochs),
        eta_min=0.85 / 0.95,
        begin=0,
        end=(0.4 * num_epochs),
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type="CosineAnnealingMomentum",
        T_max=12,
        eta_min=1,
        begin=(0.4 * num_epochs),
        end=num_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

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
    ),
    dict(
        type="ProfilerHook", 
        activity_with_cpu=True, 
        activity_with_cuda=True,
        with_stack=True,
        by_epoch=False,
        profile_times=6,
        schedule=dict(
            wait=1,
            warmup=1,
            active=1,
            repeat=1,
        ),
        json_trace_path=f"{work_dir}/profiler_trace.json",
    ),
]

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type='WandbVisBackend',
        save_dir=work_dir,
        init_kwargs=dict(
            entity="trailab",
            project="Sparse4Dv3-LC",
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
# grid sampling backprop is not deterministic
randomness = dict(seed=0, deterministic=False)

# only set for debugging
cfg = dict(
    model_wrapper_cfg=dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=True,
        detect_anomalous_params=True),
)
