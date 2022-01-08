import torch.distributed as dist

from colossalai.registry import DIST_GROUP_INITIALIZER
from colossalai.global_variables import moe_env
from .process_group_initializer import ProcessGroupInitializer
from ..parallel_mode import ParallelMode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Moemodel(ProcessGroupInitializer):
    """Model parallel initialization for MoE system.
    """

    def __init__(self, moe_model, moe_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_model = moe_model
        self.moe_data = moe_data

    def init_dist_group(self):
        """Initialize model parallel groups in moe parallel environment,
        and assign local_ranks and groups to each gpu.
        """

        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.MOE_MODEL

        for i in range(self.moe_data):
            ranks = [i * self.moe_model + j for j in range(self.moe_model)]
            group = dist.new_group(ranks)

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, ranks_in_group, mode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Moedata(ProcessGroupInitializer):
    """Data parallel initialization for MoE system.
    """

    def __init__(self, moe_model, moe_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_model = moe_model
        self.moe_data = moe_data

    def init_dist_group(self):
        """Initialize data parallel groups in moe parallel environment,
        and assign local_ranks and groups to each gpu.
        """

        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.MOE_DATA

        for i in range(self.moe_model):
            ranks = [i + j * self.moe_model for j in range(self.moe_data)]
            group = dist.new_group(ranks)

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

        return local_rank, group_world_size, process_group, ranks_in_group, mode


@DIST_GROUP_INITIALIZER.register_module
class Initializer_Moe(ProcessGroupInitializer):
    """Serves as the single entry point to MoE parallel initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_model = moe_env.model_parallel_size
        self.moe_data = moe_env.data_parallel_size
        self.model_initializer = Initializer_Moemodel(
            self.moe_model, self.moe_data, *args, **kwargs)
        self.data_initializer = Initializer_Moedata(
            self.moe_model, self.moe_data, *args, **kwargs)

    def init_dist_group(self):
        """Initializes MoE parallel communication groups.
        """

        parallel_setting = [self.model_initializer.init_dist_group(),
                            self.data_initializer.init_dist_group()]
        return parallel_setting
