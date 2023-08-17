# model settings
model = dict(
    type='CLRNet',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        start_level=1,
        num_outs=3),
    lane_head=dict(
        type='CLRHead',
        num_points=72,
        sample_points=36,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=192,
        num_fc=2,
        refine_layers=3,
        num_classes=1,
        seg_num_classes=6,
        img_size=(800, 320),
        with_seg=True,
        loss_cls=dict(
            type='SoftmaxFocalloss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=6.0
        ),
        loss_lane=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=0.5
        ),
        loss_iou=dict(
            type='LineIou_Loss',
            length=15,
            reduction='mean',
            loss_weight=2.0
        ),
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=1.0,
            bg_cls_weight=0.4,
            loss_weight=1.0,
            reduction='mean',
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='DynamicAssigner',
            distance_cost_weight=3.0,
            cls_cost=dict(type='FocalLossCost', weight=1.0),
            iou_calculator=dict(type='LineIou', length=15),
        )
    ),
    test_cfg=dict(
        score_thr=0.4,
        nms_thres=50,
        max_lanes=5
    )
)
