# dataset settings
dataset_type = 'TusimpleDataset'
data_root = 'data/tusimple'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)

img_size = (800, 320)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=True, with_seg=True, with_lane_exist=True, seg_classs_agnostic=False),
    dict(type='FixedCrop', crop=(0, 160, 1280, 720)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', affine_ratio=0.7, degrees=10, translate=.1, scale=.2, shear=0.0),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GenerateLaneLine', num_points=72, with_theta=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_lanes', 'gt_labels', 'gt_semantic_seg']),
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

evaluation = dict(interval=1)