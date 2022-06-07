import copy
import heapq

from colossalai.builder import build_model, build_layer
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
import torch.nn as nn





def count_layer_params(layers):
    """Count the number of parameters in each layer
    """
    param_counts = [0] * len(layers)
    for idx, cfg in enumerate(layers):
        layer = build_layer(cfg)
        params = filter(lambda p: p.requires_grad, layer.parameters())
        param_counts[idx] = sum(p.numel() for p in params)

    return param_counts


def build_pipeline_model_from_cfg(config,
                                  num_chunks: int = 1,
                                  partition_method: str = 'parameter',
                                  verbose: bool = False):
    """An initializer to split the model into different stages for pipeline parallelism.

    An example for the model config is shown below. The class VisionTransformerFromConfig should
    inherit colossalai.nn.model.ModelFromConfig to allow this initializer to build model from a sequence
    of layer configurations.

    ::

        model_config = dict(
            type='VisionTransformerFromConfig',
            embedding_cfg=dict(...),
            ...
        )

    Args:
        config (dict): Configuration of the model.
        num_chunks (int, optional): The number of chunks you want to have on the current stage.
            This value should be 1 in most cases unless you are using virtual pipeline parallelism.
        partition_method (str, optional): This parameter determines how you want to split your model
            layers into stages, you can set it as 'layer' or 'parameter'.
        verbose (bool, optional): Whether to print the logs.
    """
    ori_model = build_model(config)
    layers = ori_model.layers_cfg
    layer_length = len(layers)
    logger = get_dist_logger()
    if verbose:
        logger.info(f"The total length of layers is {layer_length}", ranks=[0])

    pipeline_parallel_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    method = partition_method.lower()
    # Make a partition
    if method == 'layer':
        num_layers = len(layers)
        parts = partition_uniform(num_layers, pipeline_parallel_size, num_chunks)
    elif method == 'parameter':
        param_counts = count_layer_params(layers)
        # print_rank_0(param_counts)
        parts = partition_balanced(param_counts, pipeline_parallel_size, num_chunks)
    else:
        raise ValueError("Method should be a pre-set string in [layer, parameter]")

    # Display the partition
    if verbose:
        log_str = 'Layer allocation after partitioning: \n'
        for stage in range(pipeline_parallel_size):

            num_layers = 0
            for st, ed in parts[stage]:
                num_layers += ed - st

            log_str += f'\n===== stage={stage}, layers={num_layers} =====\n'
            for st, ed in parts[stage]:
                for idx, layer in enumerate(layers[st:ed]):
                    log_str += f'\t{idx + st:2d}: {layer}\n'
        logger.info(log_str, ranks=[0])

    # Save the partition
    interval = parts[pipeline_rank]

    models = []
    for st, ed in interval:
        model = copy.deepcopy(ori_model)
        model.build_from_cfg(st, ed)
        models.append(model)

    return nn.ModuleList(models) if len(models) > 1 else models[0]


def build_pipeline_model(layers: nn.Sequential, num_chunks: int = 1, verbose: bool = False):
    """An intializer to split the model into different stages for pipeline parallelism.
    Note that `layer` must be `torch.nn.Sequential`.
    Args:
        layers (`torch.nn.Sequential`): Layers of model
        num_chunks: The number of chunks you want to have on the current stage. This value should be 1
                        in most cases unless you are using virtual pipeline parallelism.
        verbose (bool, optional): Whether to print the logs.
    """
    pipeline_parallel_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    partitions = partition_uniform(len(layers), pipeline_parallel_size, num_chunks)
    module_list = []
    for start, end in partitions[pipeline_rank]:
        module_list.append(
            nn.Sequential(*[nn.Identity() for _ in range(start)], *layers[start:end],
                          *[nn.Identity() for _ in range(len(layers) - end)]))
    if verbose:
        logger = get_dist_logger()
        logger.info(f'Total {len(layers)} layers', ranks=[0])
        for rank, part in enumerate(partitions):
            log_str = f'===== stage={rank} =====\n'
            for chunk, (start, end) in enumerate(part):
                log_str += f'===== chunk={chunk}, layer=[{start}-{end}] =====\n'
                log_str += '\n'.join([str(layer) for layer in layers[start:end]]) + '\n'
            logger.info(log_str, ranks=[0])
    return nn.ModuleList(module_list) if len(module_list) > 1 else module_list[0]
