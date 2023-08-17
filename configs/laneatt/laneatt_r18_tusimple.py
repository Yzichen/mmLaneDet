_base_ = [
    "../_base_/datasets/tusimple.py",
    "../_base_/models/laneatt_r18.py",
    "../_base_/default_runtime.py"
]

# dataset settings
dataset_type = 'TusimpleDataset'
data_root = 'data/tusimple'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=False)

img_size = (640, 360)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadLaneAnnotations', with_lane=True, with_seg=False, with_lane_exist=False, seg_classs_agnostic=False),
    dict(type='FixedCrop', crop=(0, 0, 1280, 720)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomAffine', affine_ratio=0.7, degrees=10, translate=.1, scale=.2, shear=0.0),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GenerateLaneLine', num_points=72, with_theta=False),     # !!! 注意这里不需要theta
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_lanes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixedCrop', crop=(0, 0, 1280, 720)),
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

model = dict(
    lane_head=dict(
        anchors_freq_path='./cache/tusimple_anchors_freq.pt',
        topk_anchors=1000,
    ),
    test_cfg=dict(
        score_thr=0.2,
        nms_thres=45,
        max_lanes=5
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

evaluation = dict(interval=1)

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.01)
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

total_epochs = 100
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(interval=5)
load_from=None
resume_from=None
