# dataset settings
 # TODO : 이걸로 바꾸기 'CustomDataset'  

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_type = 'CustomDataset'        # TODO Customdataset으로 dataset>coco.py를 수정

data_root = 'data'
data_category = 'paprika'
dataset_json = 'dataset.json'

data = dict(
    samples_per_gpu=1,  # batch_size
    workers_per_gpu=1, # 1? 2? ???
    train=dict(
        type=dataset_type,
        ann_file=data_root + "/" + data_category + "/" + dataset_json,    
        img_prefix=data_root  + "/" + data_category + "/",              
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "/" + data_category + "/" + dataset_json,     
        img_prefix=data_root  + "/" + data_category + "/",               
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "/" + data_category + "/" + dataset_json,     
        img_prefix=data_root  + "/" + data_category + "/",              
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])


