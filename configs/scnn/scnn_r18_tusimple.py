_base_ = [
    "../_base_/datasets/tusimple.py",
    "../_base_/models/scnn_r18.py",
    "../_base_/default_runtime.py"
]


# dataset settings
dataset_type = 'TusimpleDataset'
data_root = 'data/tusimple'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


img_size = (800, 320)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=False, with_seg=True, with_lane_exist=True, seg_classs_agnostic=False),
    dict(type='FixedCrop', crop=(0, 160, 1280, 720)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', affine_ratio=0.5, degrees=10, translate=0.0, scale=0.0, shear=0.0),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'lane_exist']),
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
        ]
    )
]

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
optimizer = dict(type='SGD', lr=0.025, weight_decay=1e-4, momentum=0.9)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Poly',
    power=0.9,
    min_lr=0,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    by_epoch=False
)

total_epochs = 100
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
evaluation = dict(interval=5)
load_from=None
resume_from=None