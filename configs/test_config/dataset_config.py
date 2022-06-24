
mode = 'test'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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

dataset_type = 'CustomDataset'       

data_root = 'data'
data_category = 'paprika'
dataset_json = 'dataset.json'

data = dict(
    samples_per_gpu=2,  # batch_size
    workers_per_gpu=1, 
    test=dict(
        type=dataset_type,
        ann_file= None,                                                          # work_dir/model_dir/dataset.json
        img_prefix=data_root  + "/test/" + data_category + "/",                 # test할 image의 dir        
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])