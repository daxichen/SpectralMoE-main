# dataset config
_base_ = [
    "../_base_/datasets/dg_gid_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/depthmoe_dinov3_mask2former_gid.py"
]
train_pipeline = [
    dict(type="LoadMultiSpectralImageFromFile"),
    dict(type="LoadAnnotationsWithPIL"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(6, 13)],
        resize_type="ResizeShortestEdge",
        max_size=1024,
    ),
    dict(type="RandomCrop", crop_size={{_base_.crop_size}}, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="MSIPhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "experts_a": embed_multi,
            "experts_b": embed_multi,
            "w_gate": embed_multi,
            "w_noise": embed_multi,
            "depthmoe.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=78760, by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=78760, val_interval=3938)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=3938, max_keep_ckpts=10
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
