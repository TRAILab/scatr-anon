use_deformable_func = True
embed_dims = 256
num_heads = 8
num_learned_groups = 2
num_learned_temp_groups = 2
num_dn_groups = 6
num_temp_dn_groups = 3
num_decoder = 6
num_single_frame_decoder = 1
decouple_attn = True
temporal = True
drop_out = 0.1
with_quality_estimation = True
tracking_threshold = 0.2
multistage_heatmap = 1  # 1 for LiDAR, 2 for fusion, False for no heatmap init
init_pq_with_heatmap = False

feat_pool = True
dup_pq_groups = True

model = dict(
    type="Sparse4D",
    use_grid_mask=True,
    freeze_img=False,
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
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        loss_weight=0.2,
    ),
    pts_bbox_head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        modality="camera",
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            num_learned_groups=num_learned_groups,
            num_learned_temp_groups=num_learned_temp_groups,
            group_selection=['topk', 'random'],
            embed_dims=embed_dims,
            anchor="_nuscenes_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=True,  # true for multiple learned groups
            # heatmap init params
            heatmap_init=init_pq_with_heatmap,
            num_heatmap_stages=multistage_heatmap,
            xy_size=(180, 180),
            nms_kernel_size=3,
            num_bbox_pool_points=7,
            feat_pool=feat_pool,
            dup_pq_groups=dup_pq_groups,
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
            num_heads=num_heads,
            batch_first=True,
            dropout=drop_out,
        )
        if temporal
        else None,
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_heads,
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
            num_groups=num_heads,
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
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=num_dn_groups,
            num_temp_dn_groups=num_temp_dn_groups,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            feat_pool=feat_pool,
            second_chance_tq=True
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
        ),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0,
        ),
        loss_heatmap_reg=dict(
            type="mmdet.L1Loss",
            loss_weight=0.25
        ),
        decoder=dict(type="SparseBox3DDecoder",
                     score_threshold=tracking_threshold),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)