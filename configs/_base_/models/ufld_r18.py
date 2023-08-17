# model settings
model = dict(
    type='UFLD',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    lane_head=dict(
        type='UFLDHead',
        in_channels=512,
        row_anchor_num=56,
        grid_num=100,
        max_num_lanes=6,
        with_seg=False,
        sync_cls_avg_factor=True,
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=1.0,
            bg_cls_weight=0.4,
            loss_weight=1.0,
            reduction='mean',
        ),
        loss_cls=dict(
            type='SoftmaxFocalloss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=1.0,
            reduction='mean',
            loss_weight=1.0
        ),
    ),
    train_cfg=None,
    test_cfg=dict(
        localization_type='rel',
        sample_y=range(64, 288, 4)
    ),
)