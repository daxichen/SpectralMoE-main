OHS_type = "PotsdamDatasetNew"
OHS_root = "data/Potsdam2Vaihingen/"
OHS_crop_size = (384, 384)
# OHS_crop_size = (896, 896)

OHS_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(
    #     type='RandomResize',
    #     scale=(2048, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=OHS_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
OHS_test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
train_OHS = dict(
    type=OHS_type,
    data_root=OHS_root,
    data_prefix=dict(
        img_path="Potsdam/image",
        seg_map_path="Potsdam/label",
    ),
    pipeline=OHS_train_pipeline,
)
val_OHS = dict(
    type=OHS_type,
    data_root=OHS_root,
    data_prefix=dict(
        img_path="Vaihingen/image",
        seg_map_path="Vaihingen/label",
    ),
    pipeline=OHS_test_pipeline,
)