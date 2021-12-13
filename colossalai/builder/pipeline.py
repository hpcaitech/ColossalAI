import copy
import heapq

from colossalai.builder import build_model, build_layer
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import set_to_cuda


def _binary_partition(weights, st, ed):
    """Returns the binary partition position of `weights`, given the start
    position `st` and the end position `ed`.

    :param weights: A python list to be binary partitioned
    :type weights: list
    :param st: the start position of the binary partition
    :type st: int
    :param ed: the end postition of the binary partition
    :type ed: int
    :return: the binary partition position of `weights`
    :rtype: int
    """
    w_sum = weights[ed - 1]
    prefix = 0
    if st > 0:
        w_sum -= weights[st - 1]
        prefix = weights[st - 1]
    minimum = float("inf")
    for idx in range(st + 1, ed):
        front = weights[idx - 1] - prefix
        diff = abs(w_sum - 2 * front)
        if diff < minimum:
            pos = idx
            minimum = diff

    return st, pos, ed


def _heap_addition(weights, intervals, add_cnt):
    """
    """
    def _heap_push(heap, st, ed):
        value = weights[ed - 1]
        if st > 0:
            value -= weights[st - 1]
        heapq.heappush(heap, (-value, st, ed))

    ret_intervals = []
    heap = []

    for st, ed in intervals:
        _heap_push(heap, st, ed)

    while add_cnt > 0:
        _, st, ed = heapq.heappop(heap)
        if ed - st == 1:
            ret_intervals.append((st, ed))
        else:
            l, m, r = _binary_partition(weights, st, ed)
            _heap_push(heap, l, m)
            _heap_push(heap, m, r)
            add_cnt -= 1

    while heap:
        _, st, ed = heapq.heappop(heap)
        ret_intervals.append((st, ed))

    ret_intervals.sort()
    return ret_intervals


def _calc_partitions(weights, value):
    prev = 0
    prefix = 0
    num_block = 0
    intervals = []

    for idx, w in enumerate(weights):
        if weights[idx] - prefix > value:
            intervals.append((prev, idx))
            prev = idx
            prefix = weights[idx - 1]
            num_block += 1

    intervals.append((prev, len(weights)))
    return num_block + 1, intervals


def _binary_search(weights, num):
    length = len(weights)
    prefix = [1 if w == 0 else w for w in weights]
    for i in range(1, length):
        prefix[i] += prefix[i - 1]

    lower_bound = max(weights)
    upper_bound = prefix[length - 1]

    while upper_bound > lower_bound:
        mid = (upper_bound + lower_bound) // 2
        number, _ = _calc_partitions(prefix, mid)
        if number <= num:
            upper_bound = mid
        else:
            lower_bound = mid + 1

    num_block, intervals = _calc_partitions(prefix, upper_bound)
    if num_block < num:
        intervals = _heap_addition(prefix, intervals, num - num_block)

    return intervals


def _partition_uniform(num_items, pipeline_parallel_size, num_chunks):
    assert num_items % num_chunks == 0, \
        "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    logger = get_dist_logger()
    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    return parts


def _partition_balanced(weights, pipeline_parallel_size, num_chunks):
    num_total = pipeline_parallel_size * num_chunks
    num_items = len(weights)
    if num_items <= num_total:
        return _partition_uniform(num_items, pipeline_parallel_size, num_chunks)

    intervals = _binary_search(weights, num_total)

    current = 0
    parts = [[] for _ in range(pipeline_parallel_size)]
    for inter in intervals:
        parts[current].append(inter)
        current = (current + 1) % pipeline_parallel_size

    return parts


class PipelineModelInitializer():
    """An intializer to split the model into different stages for pipeline parallelism.

    An example for the model config is shown below. The class VisionTransformerFromConfig should
    inherit colossalai.nn.model.ModelFromConfig to allow this initializer to build model from a sequence
    of layer configurations.

    model_config = dict(
        type='VisionTransformerFromConfig',
        embedding_cfg=dict(...),
        ...
    )

    :param config: configuration of the model
    :type config: dict
    :param num_chunks: the number of chunks you want to have on the current stage. This value should be 1
                        in most cases unless you are using virutal pipeline parallelism.
    :type num_chunks: int
    :param verbose: whether to print the logs
    :type verbose: bool

    """

    def __init__(self, config, num_chunks, verbose=False):
        self.num_chunks = num_chunks
        self.ori_model = build_model(config)
        self.layers = self.ori_model.layers_cfg
        layer_length = len(self.layers)
        self.verbose = verbose
        self._logger = get_dist_logger()
        self._logger.info(f"The total length of layers is {layer_length}", ranks=[0])

    def initialize(self, partition_method='parameter'):
        """Initialize the model object from the config passed

        :param partition_method: this parameter determines how you want to split your model layers into stages,
                                you can set it as 'layer' or 'parameter'
        :type partition_method: str
        
        """
        # Some space for initializing comunication groups
        self._interval = None
        self._partition_layers(method=partition_method)
        models = self._build()
        model = set_to_cuda(models)

        return model

    def _partition_layers(self, method):
        pipeline_parallel_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        method = method.lower()
        # Make a partition
        if method == 'layer':
            num_layers = len(self.layers)
            self.parts = _partition_uniform(num_layers, pipeline_parallel_size, self.num_chunks)
        elif method == 'parameter':
            param_counts = self._count_layer_params()
            # print_rank_0(param_counts)
            self.parts = _partition_balanced(param_counts, pipeline_parallel_size, self.num_chunks)
        else:
            raise ValueError("Method should be a pre-set string in [layer, parameter]")

        # Display the partition
        if gpc.get_global_rank() == 0 and self.verbose:
            log_str = 'Layer allocation after partitioning: \n'
            for stage in range(pipeline_parallel_size):

                num_layers = 0
                for st, ed in self.parts[stage]:
                    num_layers += ed - st

                log_str += f'\n===== stage={stage}, layers={num_layers} =====\n'
                for st, ed in self.parts[stage]:
                    for idx, layer in enumerate(self.layers[st: ed]):
                        log_str += f'\t{idx + st:2d}: {layer}\n'
            self._logger.info(log_str, ranks=[0])

        # Save the partition
        self._interval = self.parts[pipeline_rank]

    def _build(self):
        """Build model from the layer cfg according to the partition
        """
        models = []
        for st, ed in self._interval:
            model = copy.copy(self.ori_model)
            model.build_from_cfg(st, ed)
            models.append(model)

        return models

    def _count_layer_params(self):
        """Count the number of parameters in each layer
        """
        param_counts = [0] * len(self.layers)
        for idx, cfg in enumerate(self.layers):
            layer = build_layer(cfg)
            params = filter(lambda p: p.requires_grad, layer.parameters())
            param_counts[idx] = sum(p.numel() for p in params)

        return param_counts
