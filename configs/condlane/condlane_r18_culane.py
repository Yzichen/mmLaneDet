_base_ = [
    "../_base_/datasets/culane.py",
    "../_base_/models/condlane_r18.py",
    "../_base_/default_runtime.py"
]

# dataset settings
dataset_type = 'CuLaneDataset'
data_root = 'data/culane'
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)

img_size = (800, 320)
down_scale = 8
hm_down_scale = 16

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
    dict(type='GenerateCondInfo', down_scale=down_scale, hm_down_scale=hm_down_scale),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_hm', 'gt_masks']),
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
    samples_per_gpu=4,
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
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Step',
    step=[8, 14],
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    by_epoch=True
)

total_epochs = 16
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(start=1, interval=1)
load_from=None
resume_from=None