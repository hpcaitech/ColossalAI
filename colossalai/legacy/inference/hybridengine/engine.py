import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding

from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.schedule.generate import GenerateSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.base_policy import Policy

from ..pipeline.microbatch_manager import MicroBatchManager
from ..tensor_parallel.kvcache_manager import MemoryManager

PP_AXIS, TP_AXIS = 0, 1

_supported_models = [
    "LlamaForCausalLM",
]


class CaiInferEngine:
    """
    CaiInferEngine is a class that handles the pipeline parallel inference.

    Args:
        tp_size (int): the size of tensor parallelism.
        pp_size (int): the size of pipeline parallelism.
        model (`nn.Module`): the model not in pipeline style, and will be modified with `ShardFormer`.
        model_policy (`colossalai.shardformer.policies.base_policy.Policy`): the policy to shardformer model.
        micro_batch_size (int): the micro batch size.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.
        max_batch_size (int): the maximum batch size.
        max_input_len (int): the maximum input length.
        max_output_len (int): the maximum output length.

    Example:

    ```python
    from colossalai.inference import InferEngine
    from colossalai.inference.pipeline.policies import LlamaModelInferPolicy
    import colossalai
    from transformers import LlamaForCausalLM, LlamaTokenizer

    colossalai.launch_from_torch()

    model = LlamaForCausalLM.from_pretrained("your_path_to_model")
    tokenizer = LlamaTokenizer.from_pretrained("/home/lczyh/share/models/llama-7b-hf")
    # assume the model is inferred with 2 pipeline stages
    inferengine = CaiInferEngine(pp_size=2, model=model, model_policy=LlamaModelInferPolicy())

    input = ["Introduce a landmark in China ","Introduce a landmark in China "]
    data = tokenizer(input, return_tensors='pt')
    output = inferengine.inference([data.to('cuda').data])

    ```

    """

    def __init__(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        dtype: str = "fp16",
        model: nn.Module = None,
        model_policy: Policy = None,
        micro_batch_size: int = 1,
        micro_batch_buffer_size: int = None,
        max_batch_size: int = 4,
        max_input_len: int = 32,
        max_output_len: int = 32,
        verbose: bool = False,
        # TODO: implement early_stopping, and various generation options
        early_stopping: bool = False,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> None:
        assert model.__class__.__name__ in _supported_models, f"Model {model.__class__.__name__} is not supported."
        assert (
            tp_size * pp_size == dist.get_world_size()
        ), f"TP size({tp_size}) * PP size({pp_size}) should be equal to the global world size ({dist.get_world_size()})"
        assert model and model_policy, "Model with model_policy should be provided."
        assert dtype in ["fp16", "fp32", "bf16"], "dtype should be one of 'fp16', 'fp32', 'bf16'"

        assert max_batch_size <= 64, "Max batch size exceeds the constraint"
        assert max_input_len + max_output_len <= 4096, "Max length exceeds the constraint"

        # TODO: support only tensor parallel inference
        assert pp_size > 1, "Not support only tensor parallel inference."
        self.pp_size = pp_size
        self.tp_size = tp_size

        if dtype == "fp16":
            self.dtype = torch.float16
            model.half()
        elif dtype == "bf16":
            self.dtype = torch.bfloat16
            model.to(torch.bfloat16)
        else:
            self.dtype = torch.float32

        # Init pg mesh
        pg_mesh = ProcessGroupMesh(pp_size, tp_size)

        stage_manager = None
        if pp_size > 1:
            stage_manager = PipelineStageManager(pg_mesh, PP_AXIS, True)
            self.cache_manager_list = [
                self._init_manager(model, max_batch_size, max_input_len, max_output_len)
                for _ in range(micro_batch_buffer_size or pp_size)
            ]
            self.mb_manager = MicroBatchManager(
                stage_manager.stage,
                micro_batch_size,
                micro_batch_buffer_size or pp_size,
                max_input_len,
                max_output_len,
                self.cache_manager_list,
            )
            self.verbose = verbose
            self.schedule = GenerateSchedule(stage_manager, self.mb_manager, verbose)

        self.model = self._shardformer(model, model_policy, stage_manager, pg_mesh.get_group_along_axis(TP_AXIS))

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
        ), f"Only accept BatchEncoding or dict as input, but got {input_list.__class__.__name__}."
        if isinstance(input_list, BatchEncoding):
            input_list = input_list.data
        out, timestamp = self.schedule.generate_step(self.model, iter([input_list]))
        if self.verbose:
            return out, timestamp
        else:
            return out

    def _shardformer(self, model, model_policy, stage_manager, tp_group):
        shardconfig = ShardConfig(
            tensor_parallel_process_group=tp_group,
            pipeline_stage_manager=stage_manager,
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

    def _init_manager(self, model, max_batch_size: int, max_input_len: int, max_output_len: int) -> None:
        max_total_token_num = max_batch_size * (max_input_len + max_output_len)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        head_num = model.config.num_attention_heads
        num_hidden_layers = (
            model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.config.num_layers
        )
        layer_num = num_hidden_layers // self.pp_size

        cache_manager = MemoryManager(max_total_token_num, self.dtype, head_num, head_dim, layer_num)
        return cache_manager
