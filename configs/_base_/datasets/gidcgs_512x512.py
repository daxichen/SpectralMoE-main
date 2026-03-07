OHS_type = "GID_Dataset"
OHS_root = "data/GID_TG/"
# OHS_root = "data/GID_Sentinel2/"

OHS_crop_size = (512, 512)
# OHS_crop_size = (384, 384)
OHS_train_pipeline = [
    # dict(type="LoadSingleRSImageFromFile"),
    dict(type="LoadMultiSpectralImageFromFile"),
    dict(type="LoadAnnotationsWithPIL"),
    # dict(type="Resize", scale=(1280, 720)),
    # dict(type="RandomCrop", crop_size=OHS_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    # dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
OHS_test_pipeline = [
    # dict(type="LoadSingleRSImageFromFile"),
    dict(type="LoadMultiSpectralImageFromFile"),
    # dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotationsWithPIL"),
    dict(type="PackSegInputs"),
]
train_OHS = dict(
    type=OHS_type,
    data_root=OHS_root,
    data_prefix=dict(
        img_path="source_dir/image",
        seg_map_path="source_dir/label",
        # img_path="source_dir\\image",
        # seg_map_path="source_dir\\label",
        # seg_map_path="source_dir/label_merged",
    ),
    pipeline=OHS_train_pipeline,
)
val_OHS = dict(
    type=OHS_type,
    data_root=OHS_root,
    data_prefix=dict(
        img_path="target_dir/image",
        seg_map_path="target_dir/label",
        # img_path="target_dir\\image",
        # seg_map_path="target_dir\\label",
        # seg_map_path="target_dir/label_merged",
    ),
    pipeline=OHS_test_pipeline,
)