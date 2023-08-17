_base_ = [
    "../_base_/datasets/tusimple.py",
    "../_base_/models/bezier_r18.py",
    "../_base_/default_runtime.py"
]

# dataset settings
dataset_type = 'TusimpleDataset'
data_root = 'data/tusimple'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_size = (800, 320)
order = 4

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=True, with_seg=True, with_lane_exist=False, seg_classs_agnostic=False),
    dict(type='Lanes2ControlPoints', order=order),
    dict(type='FixedCrop', crop=(0, 160, 1280, 720)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', affine_ratio=0.7, degrees=10, translate=0.1, scale=0.2, shear=0.0),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GenerateBezierInfo', order=order, num_sample_points=100),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_control_points', 'gt_labels', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixedCrop', crop=(0, 160, 1280, 720)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
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

model = dict(
    lane_head=dict(
        with_seg=False
    )
)


# SPLIT_FILES = {
#     'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
#     'train': ['label_data_0313.json', 'label_data_0601.json'],
#     'val': ['label_data_0531.json'],
#     'test': ['test_label.json'],
# }


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
optimizer = dict(
    type='Adam',
    lr=1e-3,
    paramwise_cfg=dict(
        custom_keys={
            'conv_offset': dict(lr_mult=0.1),
        }),
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    by_epoch=True
)

total_epochs = 400
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
evaluation = dict(interval=10)
load_from=None
resume_from=None