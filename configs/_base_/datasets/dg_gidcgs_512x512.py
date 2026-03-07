_base_ = [
    "./gidcgs_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_OHS}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_OHS}},
        ],
    ),
)
test_dataloader = val_dataloader
# val_evaluator = dict(
#     type="IoUMetric", iou_metrics=["mFscore"]
# )
# test_evaluator= dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])

val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["GID_TG/"],
    mean_used_keys=["GID_TG/"],
    # dataset_keys=["GID_Sentinel2/"],
    # mean_used_keys=["GID_Sentinel2/"],
)

test_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU"],
    format_only=False,
    output_dir="./work_dirs/results_FMoLTE_DOFA_GID_CGS",
)