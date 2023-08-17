# model settings
model = dict(
    type='RESA_Detector',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        strides=(1, 2, 1, 1),   # 利用dilate=2 代替 stride=2
        dilations=(1, 1, 2, 2),
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    aggregator=dict(
        type='RESA',
        in_channels=512,
        mid_channels=None,
        out_channels=128,
        kernel_size=9,
        directions=['d', 'u', 'r', 'l'],
        alpha=2.0,
        num_iters=4,
        img_size=(800, 320),
        stride=8
    ),
    lane_head=dict(
        type='RESAHead',
        num_points=64,
        max_num_lanes=6,
        img_size=(800, 320),
        stride=8,
        in_channels=128,
        fc_hidden_dim=128,
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=1.0,
            bg_cls_weight=0.4,
            loss_weight=1.0,
            reduction='mean',
        ),
        loss_exist=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.1
        ),
        with_exist=False,
    ),
    train_cfg=None,
    test_cfg=dict(
        exist_score_thr=0.5,
        seg_score_thr=0.6,
        min_num_lanes=5,
        sample_y=range(710, 150, -10)
    ),
)