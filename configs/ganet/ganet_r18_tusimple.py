_base_ = [
    "../_base_/datasets/tusimple.py",
    "../_base_/models/ganet_r18.py",
    "../_base_/default_runtime.py"
]

dataset_type = 'TusimpleDataset'
data_root = 'data/tusimple'
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)

img_size = (800, 320)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=True, with_seg=True, with_lane_exist=True, seg_classs_agnostic=False),
    dict(type='FixedCrop', crop=(0, 160, 1280, 720)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', affine_ratio=0.7, degrees=10, translate=.1, scale=.2, shear=0.0),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GenerateGAInfo',
         radius=2,
         fpn_cfg=dict(
             hm_idx=0,
             fpn_down_scale=[8, 16, 32],
             sample_per_lane=[41, 21, 11],
        )),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_targets']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixedCrop', crop=(0, 160, 1280, 720)),
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
        kpt_thr=0.4,
        cluster_by_center_thr=4,
        hm_down_scale=8
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        anno_files=['annos_6/train_val.json'],
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        anno_files=['annos_6/test.json'],
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        anno_files=['annos_6/test.json'],
        pipeline=test_pipeline,
        test_mode=True)
)

# optimizer
optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='Poly',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10, # LR used at the beginning of warmup equals to warmup_ratio * initial_lr
    min_lr=1e-5,
    by_epoch=True
)

total_epochs = 70
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
evaluation = dict(interval=5, save_best='F1_score', greater_keys=['F1_score'])
load_from=None
resume_from=None
