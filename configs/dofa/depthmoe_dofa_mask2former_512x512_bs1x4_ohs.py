# dataset config
_base_ = [
    "../_base_/datasets/dg_whu_ohs_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/depthmoe_dofa_mask2former_ohs.py"
]
train_pipeline = [
    dict(type="LoadHyperRSImageFromFile"),
    dict(type="LoadAnnotationsForTIF"),
    dict(type="RandomFlip", prob=0.5),
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
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=10945, by_epoch=False)
]


# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=10945, val_interval=547)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=547, max_keep_ckpts=20
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
