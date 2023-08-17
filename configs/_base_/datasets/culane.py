# dataset settings
dataset_type = 'CuLaneDataset'
data_root = 'data/culane'
img_norm_cfg = dict(
    mean=[255, 255, 255], std=[1, 1, 1], to_rgb=False)

img_size = (800, 320)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadLaneAnnotations', with_lane=True, with_seg=True, seg_classs_agnostic=False),
#     dict(type='FixedCrop', crop=(0, 270, 1640, 590)),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='RandomAffine', affine_ratio=0.7, degrees=10, translate=.1, scale=.2, shear=0.0),
#     dict(type='Resize', img_scale=img_size, keep_ratio=False),
#     # dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='GenerateLaneLine', num_points=72, with_theta=True),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_lanes', 'gt_labels', 'gt_semantic_seg']),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=True, with_seg=True, with_lane_exist=True, seg_classs_agnostic=False),
    dict(type='FixedCrop', crop=(0, 270, 1640, 590)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='albumentation', pipelines=[
        dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
        dict(type='ChannelShuffle', p=0.1),
        dict(
            type='OneOf',
            transforms=[
                dict(
                    type='RGBShift',
                    r_shift_limit=10,
                    g_shift_limit=10,
                    b_shift_limit=10,
                    p=1.0),
                dict(
                    type='HueSaturationValue',
                    hue_shift_limit=(-10, 10),
                    sat_shift_limit=(-15, 15),
                    val_shift_limit=(-10, 10),
                    p=1.0),
            ],
            p=0.7),
        dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
        dict(
            type='OneOf',
            transforms=[
                dict(type='Blur', blur_limit=3, p=1.0),
                dict(type='MedianBlur', blur_limit=3, p=1.0)
            ],
            p=0.2),
        dict(type='RandomBrightness', limit=0.2, p=0.6),
        dict(
            type='ShiftScaleRotate',
            shift_limit=0.1,
            scale_limit=(-0.2, 0.2),
            rotate_limit=10,
            border_mode=0,
            p=0.6),
    ]),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GenerateLaneLine', num_points=72, with_theta=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_lanes', 'gt_labels', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixedCrop', crop=(0, 270, 1640, 590)),
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