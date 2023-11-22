import copy

from colossalai.shardformer import ShardConfig, ShardFormer


def build_model(
    model_fn,
    enable_fused_normalization=False,
    enable_tensor_parallelism=False,
    enable_flash_attention=False,
    enable_jit_fused=False,
):
    # create new model
    org_model = model_fn()

    # shard model
    shard_config = ShardConfig(
        enable_fused_normalization=enable_fused_normalization,
        enable_tensor_parallelism=enable_tensor_parallelism,
        enable_flash_attention=enable_flash_attention,
        enable_jit_fused=enable_jit_fused,
    )
    model_copy = copy.deepcopy(org_model)
    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, shared_params = shard_former.optimize(model_copy)
    return org_model.cuda(), sharded_model.cuda()


def run_infer(original_model, sharded_model, data_gen_fn, output_transform_fn):
    # prepare input
    data = data_gen_fn()
    data = {k: v.cuda() for k, v in data.items()}
    # run forward
    org_output = original_model(**data)
    org_output = output_transform_fn(org_output)

    shard_output = sharded_model(**data)
    shard_output = output_transform_fn(shard_output)

    return org_output, shard_output
