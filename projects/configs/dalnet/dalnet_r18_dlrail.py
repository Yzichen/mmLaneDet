_base_ = [
    "../../../configs/_base_/datasets/dlrail.py",
    "../../../configs/_base_/default_runtime.py"
]

plugin = True
plugin_dir = 'projects/mmlane_plugin/'

dataset_type = 'RailDataset'
data_root = 'data/DlRail'

img_size = (800, 320)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=True, with_seg=True, with_lane_exist=False, seg_classs_agnostic=False),
    dict(type='FixedCrop', crop=(0, 400, 1920, 1080)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', affine_ratio=0.7, degrees=10, translate=.1, scale=.2, shear=0.0),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GenerateLaneLineV2', num_points=72, with_theta=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_lanes', 'gt_labels', 'start_points', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixedCrop', crop=(0, 400, 1920, 1080)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# model settings
model = dict(
    type='DALNet',
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
    aggregator=dict(
        type='PyramidPoolingModule',
        in_channels=512,
        out_channels=64,
        sizes=(1, 2, 3, 6)
    ),
    # aggregator=dict(
    #     type='TransEncoderModule',
    #     attn_in_dims=[512, 64],
    #     attn_out_dims=[64, 64],
    #     strides=[1, 1],
    #     ratios=[4, 4],
    #     pos_shape=(1, 10, 25),
    # ),
    neck=dict(
        type='FPNV1',
        in_channels=[128, 256, 64],
        out_channels=64,
        num_outs=2,
        out_ids=[0, 1],     # 取1/8尺度 和 1/16尺度
    ),
    lane_head=dict(
        type='DALNetHead',
        in_channel=64,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_points=72,
        sample_points=36,
        img_size=(800, 320),
        num_stages=1,
        num_classes=1,
        seg_num_classes=2,
        hm_scale=16,
        radius=2,
        hm_radius=2,
        max_sample=5,
        kernel_size=5,
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_hm_reg=dict(
            type='L1Loss',
            reduction='mean',
            loss_weight=10.0
        ),
        loss_hm_offset=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_lane=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='mean',
            loss_weight=0.3    # 0.1
        ),
        loss_iou=dict(
            type='LineIou_Loss',
            length=15,
            reduction='mean',
            loss_weight=6.0     # 2.0
        ),
        with_seg=False,
        share=True,
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(
        hm_thr=0.4,
        nms_thr=2,
        max_lanes=2
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        anno_files=['list/train_gt.txt'],
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        anno_files=['list/test.txt'],
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        anno_files=['list/test.txt'],
        pipeline=test_pipeline,
        test_mode=True)
)

# optimizer
optimizer = dict(type='AdamW', lr=1.0e-3, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    # by_epoch=False
    )
total_epochs = 70
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(interval=2, start=50)
load_from=None
resume_from=None
