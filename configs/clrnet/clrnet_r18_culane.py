_base_ = [
    "../_base_/datasets/culane.py",
    "../_base_/models/clrnet_r18.py",
    "../_base_/default_runtime.py"
]


model = dict(
    lane_head=dict(
        seg_num_classes=4,
        sync_cls_avg_factor=True,
        with_seg=True,
        loss_cls=dict(
            type='SoftmaxFocalloss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=2.0
        ),
        loss_lane=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=0.2
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
            loss_weight=3.0
        ),
    ),
    test_cfg=dict(
        score_thr=0.4,
        nms_thres=50,
        max_lanes=4
    )
)

# optimizer
optimizer = dict(type='AdamW', lr=0.8e-3, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    by_epoch=False
)
total_epochs = 15
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
evaluation = dict(start=12, interval=1)
load_from=None
resume_from=None