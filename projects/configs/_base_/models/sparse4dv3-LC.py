_base_ = [
    "./sparse4dv3_temporal_SECOND.py",
]

multistage_heatmap = 2
embed_dims = {{_base_.embed_dims}}
model = dict(
    freeze_pts=True,
    freeze_fusion=False,
    freeze_img=True,
    init_cfg=[
        dict(type="Pretrained", checkpoint="ckpts/focalformer3d_converted/DeformFormer3D_C_R50_ep20_mAP300_NDS363.pth"),
        dict(type="Pretrained", checkpoint="ckpts/focalformer3d_converted/DeformFormer3D_L_iterimg_ep20_mAP655_NDS707.pth"),
    ],
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_fusion_layer=dict(
        num_layers=multistage_heatmap,
        multistage_heatmap=multistage_heatmap,
        cam_lss=True,
        iterbev='bevfusion',
        iter_bev_cam=True,
        input_img=True,
    ),
    pts_bbox_head=dict(
        reuse_first_heatmap=False,
        instance_bank=dict(
            num_heatmap_stages=multistage_heatmap,
        ),
        anchor_encoder=dict(
            output_fc=True,
            output_dim=embed_dims,
        )
    )
)