import torch
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding

from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.schedule.generate import GenerateSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.base_policy import Policy

from ..tensor_parallel.kvcache_manager import MemoryManager
from .microbatch_manager import MicroBatchManager


class PPInferEngine:
    """
    PPInferEngine is a class that handles the pipeline parallel inference.

    Args:
        pp_size (int): the number of pipeline stages.
        pp_model (`nn.Module`): the model already in pipeline parallelism style.
        model (`nn.Module`): the model not in pipeline style, and will be modified with `ShardFormer`.
        model_policy (`colossalai.shardformer.policies.base_policy.Policy`): the policy to shardformer model.
        micro_batch_size (int): the micro batch size.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.
        new_length (int): the new length of the input sequence.
        early_stopping (bool): whether to stop early.
        max_batch_size (int): the maximum batch size.
        max_input_len (int): the maximum input length.
        max_output_len (int): the maximum output length.

    Example:

    ```python
    from colossalai.inference import PPInferEngine
    from colossalai.inference.pipeline.policies import LlamaModelInferPolicy
    import colossalai
    from transformers import LlamaForCausalLM, LlamaTokenizer

    colossalai.launch_from_torch(config={})

    model = LlamaForCausalLM.from_pretrained("your_path_to_model")
    tokenizer = LlamaTokenizer.from_pretrained("/home/lczyh/share/models/llama-7b-hf")
    # assume the model is infered with 2 pipeline stages
    inferengine = PPInferEngine(pp_size=2, model=model, model_policy=LlamaModelInferPolicy(), new_length=8)

    input = ["Introduce a landmark in China ","Introduce a landmark in China "]
    data = tokenizer(input, return_tensors='pt')
    output = inferengine.inference([data.to('cuda').data])

    ```

    """

    def __init__(
        self,
        pp_size: int,
        dtype: str = "fp16",
        pp_model: nn.Module = None,
        model: nn.Module = None,
        model_policy: Policy = None,
        new_length: int = 32,
        micro_batch_size: int = 1,
        micro_batch_buffer_size: int = None,
        max_batch_size: int = 4,
        max_input_len: int = 32,
        max_output_len: int = 32,
        verbose: bool = False,
        # TODO: implement early_stopping, and various gerneration options
        early_stopping: bool = False,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> None:
        assert pp_model or (model and model_policy), "Either pp_model or model with model_policy should be provided."
        assert dtype in ["fp16", "fp32", "bf16"], "dtype should be one of 'fp16', 'fp32', 'bf16'"

        max_output_len = max(max_output_len, max_input_len + new_length)

        self.pp_size = pp_size
        if dtype == "fp16":
            self.dtype = torch.float16
            model.half()
        elif dtype == "bf16":
            self.dtype = torch.bfloat16
            model.to(torch.bfloat16)
        else:
            self.dtype = torch.float32
        self.pg_mesh = ProcessGroupMesh(pp_size)
        self.stage_manager = PipelineStageManager(self.pg_mesh, 0, True)
        self.model = pp_model or self._shardformer(model, model_policy)
        self.cache_manager_list = [
            self._init_manager(max_batch_size, max_input_len, max_output_len)
            for _ in range(micro_batch_buffer_size or pp_size)
        ]
        self.mb_manager = MicroBatchManager(
            self.stage_manager.stage,
            new_length,
            micro_batch_size,
            micro_batch_buffer_size or pp_size,
            max_input_len,
            max_output_len,
            self.cache_manager_list,
        )
        self.verbose = verbose
        self.schedule = GenerateSchedule(self.stage_manager, self.mb_manager, verbose)

    def inference(self, input_list):
        """
        Args:
            input_list (list): a list of input data, each element is a `BatchEncoding` or `dict`.

        Returns:
            out (list): a list of output data, each element is a list of token.
            timestamp (float): the time cost of the inference, only return when verbose is `True`.
        """
        assert isinstance(
            input_list, (BatchEncoding, dict)
        ), f"Only accept BatchEncoding or dict as input, but get {input_list.__class__.__name__}."
        if isinstance(input_list, BatchEncoding):
            input_list = input_list.data
        out, timestamp = self.schedule.generate_step(self.model, iter([input_list]))
        if self.verbose:
            return out, timestamp
        else:
            return out

    def _shardformer(self, model, model_policy):
        shardconfig = ShardConfig(
            tensor_parallel_process_group=None,
            pipeline_stage_manager=self.stage_manager,
            enable_tensor_parallelism=False,
            enable_fused_normalization=False,
            enable_all_optimization=False,
            enable_flash_attention=False,
            enable_jit_fused=False,
            enable_sequence_parallelism=False,
        )
        shardformer = ShardFormer(shard_config=shardconfig)
        shard_model, _ = shardformer.optimize(model, model_policy)
        return shard_model.cuda()

    def _init_manager(self, max_batch_size: int, max_input_len: int, max_output_len: int) -> None:
        max_total_token_num = max_batch_size * (max_input_len + max_output_len)
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        head_num = self.model.config.num_attention_heads
        num_hidden_layers = (
            self.model.config.num_hidden_layers
            if hasattr(self.model.config, "num_hidden_layers")
            else self.model.config.num_layers
        )
        layer_num = num_hidden_layers // self.pp_size

        cache_manager = MemoryManager(max_total_token_num, self.dtype, head_num, head_dim, layer_num)
        return cache_manager
