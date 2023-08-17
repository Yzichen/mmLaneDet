# model settings
model = dict(
    type='CondLane',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    aggregator=dict(
        type='TransEncoderModule',
        attn_in_dims=[512, 256],
        attn_out_dims=[256, 256],
        strides=[1, 1],
        ratios=[4, 4],
        pos_shape=(1, 10, 25),
    ),
    neck=dict(
        type='FPNV1',
        in_channels=[128, 256, 256],
        out_channels=64,
        num_outs=2,
        out_ids=[0, 1],     # 取1/8尺度 和 1/16尺度
        ),
    lane_head=dict(
        type='CondLaneHead',
        heads_dict={
            'hm': {'out_channels': 1, 'num_conv': 2},
        },
        in_channels=64,
        num_classes=1,
        shared_branch_mid_channels=64,
        shared_branch_out_channels=64,
        shared_branch_num_conv=3,
        disable_coords=False,
        cond_head_num_layers=1,
        compute_locations_pre=True,
        mask_shape=(40, 100),
        with_offset=True,
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_location=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_offset=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=0.4
        ),
        loss_range=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='mean',
            loss_weight=1.0
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        hm_thr=0.5,
        nms_thr=4,
        down_scale=8,
    )
)
