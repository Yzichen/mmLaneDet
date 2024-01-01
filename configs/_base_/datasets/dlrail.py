# dataset settings
dataset_type = 'RailDataset'
data_root = 'data/DlRail'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_size = (800, 320)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=True, with_seg=True, seg_classs_agnostic=False),
    dict(type='FixedCrop', crop=(0, 400, 1920, 1080)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', affine_ratio=0.7, degrees=10, translate=.1, scale=.2, shear=0.0),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GenerateLaneLine', num_points=72, with_theta=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_lanes', 'gt_labels', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixedCrop', crop=(0, 400, 1920, 1080)),
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
        ])
]

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

evaluation = dict(interval=1)