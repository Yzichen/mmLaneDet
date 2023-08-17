# model settings
model = dict(
    type='GANet',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    aggregator=dict(
        type='TransEncoderModule',
        attn_in_dims=[512, 64],
        attn_out_dims=[64, 64],
        strides=[1, 1],
        ratios=[4, 4],
        pos_shape=(1, 10, 25),
    ),
    neck=dict(
        type='DeformFPN',
        in_channels=[128, 256, 64],
        out_channels=64,
        start_level=0,
        num_outs=1,
        out_ids=[0],    # 1/8
        dcn_only_cls=True,
    ),
    lane_head=dict(
        type='GANetHead',
        in_channels=64,
        num_classes=1,
        hm_idx=0,
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_kp_offset=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=1.0
        ),
        loss_sp_offset=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=0.5
        ),
        loss_aux=dict(
            type='SmoothL1Loss',
            beta=1.0/9.0,
            reduction='mean',
            loss_weight=0.2
        )
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(
        root_thr=1.0,
        kpt_thr=0.1,    # 0.3
        cluster_by_center_thr=4,
        hm_down_scale=8
    )
)
