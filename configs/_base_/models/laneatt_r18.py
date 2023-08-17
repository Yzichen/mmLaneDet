# model settings
model = dict(
    type='LaneATT',
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
    lane_head=dict(
        type='LaneATTHead',
        in_channels=512,
        num_points=72,
        img_size=(800, 320),
        stride=32,
        anchor_feat_channels=64,
        anchors_freq_path=None,
        return_attention_matrix=True,
        num_classes=1,
        loss_cls=dict(
            type='SoftmaxFocalloss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=10.0
        ),
        loss_reg=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=1.0
        ),
        sync_cls_avg_factor=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        score_thr=0.0,
        nms_thres=15.,
        max_lanes=3000,
    ),
    test_cfg=dict(
        score_thr=0.2,
        nms_thres=45,
        max_lanes=5
    )
)
