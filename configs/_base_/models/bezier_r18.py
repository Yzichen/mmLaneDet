# model settings
order = 3

model = dict(
    type='BezierLaneNet',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    dilated_blocks=dict(
        in_channels=256,
        mid_channels=64,
        dilations=[4, 8]
    ),
    aggregator=dict(
        type='FeatureFlipFusion',
        channels=256
    ),
    lane_head=dict(
        type='BezierHead',
        in_channels=256,
        branch_channels=256,
        num_proj_layers=2,
        feature_size=(20, 50),
        order=order,
        with_seg=True,
        num_classes=1,
        seg_num_classes=1,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            class_weight=1.0 / 0.4,
            reduction='mean',
            loss_weight=0.1
        ),
        loss_reg=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=1.0,
        ),
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            ignore_index=255,
            class_weight=1.0 / 0.4,
            loss_weight=0.75,
            reduction='mean',
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='BezierHungarianAssigner',
            order=order,
            num_sample_points=100,
            alpha=0.8
        )
    ),
    test_cfg=dict(
        score_thr=0.4,
        window_size=0,
        max_lanes=5,
        num_sample_points=50,
    )
)
