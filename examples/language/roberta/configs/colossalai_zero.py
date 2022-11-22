from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.nn.optimizer import FusedAdam

# fp16 = dict(
#     mode=AMP_TYPE.TORCH,
# )

# seed = 2
zero = dict(model_config=dict(shard_strategy=TensorShardStrategy(),
                              reduce_scatter_bucket_size_mb=25,
                              fp32_reduce_scatter=False,
                              tensor_placement_policy="cuda",
                              gradient_predivide_factor=1.0,
                              reuse_fp16_shard=False),
            optimizer_config=dict(gpu_margin_mem_ratio=0.8,
                                  initial_scale=2**5,
                                  min_scale=1,
                                  growth_factor=2,
                                  backoff_factor=0.5,
                                  growth_interval=1000,
                                  hysteresis=2,
                                  max_scale=2**32))

# gradient_accumulation = 4
clip_grad_norm = 1.0
optimizer = dict(
    type=FusedAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

# 64433