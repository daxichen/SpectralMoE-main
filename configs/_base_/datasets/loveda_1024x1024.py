OHS_type = "LoveDADatasetNew"
OHS_root = "data/LoveDA/Rural2Urban/"
# OHS_root = "data/loveda_test/target_dir/"
OHS_crop_size = (512, 512)
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
    # dict(type='Pad', size_divisor=16), # ceshi feishujuji tuxiang
    dict(type='PackSegInputs')
]
train_OHS = dict(
    type=OHS_type,
    data_root=OHS_root,
    data_prefix=dict(
        img_path="Train/Rural/image",
        seg_map_path="Train/Rural/label",
    ),
    pipeline=OHS_train_pipeline,
)
val_OHS = dict(
    type=OHS_type,
    data_root=OHS_root,
    data_prefix=dict(
        img_path="Val/Urban/image",
        # img_path="image",
        seg_map_path="Val/Urban/label",
    ),
    pipeline=OHS_test_pipeline,
)