from typing import List, Tuple, Union

import rpyc
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.config import InferenceConfig, InputMetaData
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.policy import (
    NoPaddingBaichuanModelInferPolicy,
    NoPaddingLlamaModelInferPolicy,
    model_policy_map,
)
from colossalai.inference.sampler import search_tokens
from colossalai.inference.utils import get_model_size, has_index_file
from colossalai.interface import ModelWrapper
from colossalai.lazy import LazyInitContext
from colossalai.logging import get_dist_logger
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.base_policy import Policy

PP_AXIS, TP_AXIS = 0, 1

_SUPPORTED_MODELS = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "BaichuanForCausalLM": AutoModelForCausalLM,
}

_SUPPORTED_MODEL_POLICIES = {
    "NoPaddingLlamaModelInferPolicy": NoPaddingLlamaModelInferPolicy,
    "NoPaddingBaichuanModelInferPolicy": NoPaddingBaichuanModelInferPolicy,
}

logger = get_dist_logger(__name__)


class rpcWorkerService(rpyc.Service):
    """
    Execute the computation tasks and manage its own kv cache

    Func with prefix `exposed_` will be invoked by client.
    """

    def exposed_init_dist_env(self, rank, world_size, master_address, master_port):
        logger.info(f"init process group for rank {rank}")
        colossalai.launch(rank=rank, world_size=world_size, port=master_port, host=master_address)
        logger.info(f"init process group done for rank {rank}")

    def exposed_init_model(
        self, inference_config_param: dict, model_or_path: Union[nn.Module, str], model_policy_param: str = None
    ):
        assert dist.is_initialized(), "invoke init_dist_env first please!"

        self.inference_config = InferenceConfig.from_rpc_param(inference_config_param)
        model_policy = _SUPPORTED_MODEL_POLICIES[model_policy_param]() if model_policy_param else None

        self.dtype = self.inference_config.dtype
        self.verbose = True

        self._init_model(model_or_path, model_policy)
        self._init_fd_tensor()
        self._init_output_tensor()
        logger.info(f"init model done for rank {dist.get_rank()}")

    def exposed_init_cache(self, alloc_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]):
        """Initialize the physical cache on the device.

        For each layer of the model, we allocate two tensors for key and value respectively,
        with shape of [num_blocks, num_kv_heads, block_size, head_size]
        """
        kalloc_shape, valloc_shape = alloc_shape
        num_layers = self.model_config.num_hidden_layers

        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        for _ in range(num_layers):
            self.k_cache.append(
                torch.zeros(
                    kalloc_shape,
                    dtype=self.inference_config.kv_cache_dtype,
                    device=get_accelerator().get_current_device(),
                )
            )
            self.v_cache.append(
                torch.zeros(
                    valloc_shape,
                    dtype=self.inference_config.kv_cache_dtype,
                    device=get_accelerator().get_current_device(),
                )
            )
        logger.info("physical cache init over")

    def exposed_execute_model_forward(
        self, input_token_ids_param: List[int], input_meta_data_param: dict, generation_config_param: dict
    ):
        # prepare the data for model forward
        input_meta_data = InputMetaData.from_rpc_param(input_meta_data_param)
        input_meta_data.fd_inter_tensor = self.fd_inter_tensor
        if input_meta_data.is_prompts:
            n_tokens = input_meta_data.sequence_lengths.sum().item()
        else:
            n_tokens = input_meta_data.batch_size
        input_token_ids = torch.tensor(input_token_ids_param, dtype=torch.int, device=self.device)

        # execute the model
        logits = self.model(
            input_token_ids,
            self.output_tensor[:n_tokens],
            input_meta_data,
            self.k_cache,
            self.v_cache,
        )

        # sampler
        if self.inference_config.pad_input:
            logits = logits[:, -1, :]
        next_tokens = search_tokens(
            generation_config_param,
            logits,
            input_meta_data.is_prompts,
            input_meta_data.batch_token_ids,
        )

        # return the tokens generated to scheduler
        return next_tokens.tolist()

    def _init_output_tensor(self):
        alloc_shape = (
            self.inference_config.max_batch_size
            * (self.inference_config.max_input_len + self.inference_config.max_output_len),
            self.model_config.hidden_size // self.inference_config.tp_size,
        )
        self.output_tensor = torch.zeros(alloc_shape, dtype=self.dtype, device=self.device)

    def _init_fd_tensor(self):
        fd_inter_tensor = FDIntermTensors()

        if fd_inter_tensor._tensors_initialized:
            fd_inter_tensor._reset()

        # For Spec-Dec, process the speculated tokens plus the token in the last step for each seq
        max_n_tokens = self.inference_config.max_batch_size
        max_n_tokens *= self.inference_config.max_n_spec_tokens + 1

        inference_config = self.inference_config
        kv_max_split_num = (
            inference_config.max_input_len + inference_config.max_output_len + inference_config.block_size - 1
        ) // inference_config.block_size
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        fd_inter_tensor.initialize(
            max_batch_size=max_n_tokens,
            num_attn_heads=self.model_config.num_attention_heads // self.inference_config.tp_size,
            kv_max_split_num=kv_max_split_num,
            head_dim=head_dim,
            dtype=self.dtype,
            device=get_accelerator().get_current_device(),
        )

        self.fd_inter_tensor = fd_inter_tensor

    def _init_model(self, model_or_path: Union[nn.Module, str], model_policy: Policy = None):
        """
        Shard model or/and Load weight

        Shard model: When we set tp_size > 1, we will shard the model by given model_policy.
        Load Weight: If we pass a local model path, we will load the model weight by checkpoint_io. If it is a remote-transformer url, we will use `AutoModel.from_pretrained` api of transformers lib

        Args:
            model_or_path Union[nn.Module, str]: path to the checkpoint or model of transformer format.
            model_policy (Policy): the policy to replace the model
        """

        pretrained_path = None
        if isinstance(model_or_path, str):
            import colossalai.interface.pretrained as pretrained_utils

            try:
                hf_config = AutoConfig.from_pretrained(model_or_path, trust_remote_code=True, torch_dtype=self.dtype)
                arch = getattr(hf_config, "architectures")[0]
                if arch is "BaichuanForCausalLM":
                    self.logger.warning(
                        "Attention ! We use lazy init by default, which could be faster for model loading. For baichuan model, the output maybe have a slight difference with transformers"
                    )
                ctx = LazyInitContext(default_device="cuda")
                with ctx:
                    model = _SUPPORTED_MODELS[arch].from_pretrained(
                        model_or_path, trust_remote_code=True, torch_dtype=self.dtype
                    )
                pretrained_path = pretrained_utils.get_pretrained_path(model)
            except Exception as e:
                logger.error(
                    f"An exception occurred during loading model: {e}, model should be loaded by transformers\n"
                )
        else:
            model = model_or_path

        self.model_config = model.config

        torch.cuda.empty_cache()
        init_gpu_memory = torch.cuda.mem_get_info()[0]

        self.device = get_accelerator().get_current_device()
        torch.cuda.set_device(self.device)
        if self.verbose:
            logger.info(f"the device is {self.device}")

        model = model.to(dtype=self.dtype, non_blocking=False).eval()

        if self.verbose:
            logger.info(
                f"Before the shard, Rank: [{dist.get_rank()}], model size: {get_model_size(model)} GB, model's device is: {model.device}"
            )

        if model_policy is None:
            if self.inference_config.pad_input:
                model_type = "padding_" + self.model_config.model_type
            else:
                model_type = "nopadding_" + self.model_config.model_type
            model_policy = model_policy_map[model_type]()

        pg_mesh = ProcessGroupMesh(self.inference_config.pp_size, self.inference_config.tp_size)
        tp_group = pg_mesh.get_group_along_axis(TP_AXIS)

        self.model = self._shardformer(
            model,
            model_policy,
            None,
            tp_group=tp_group,
        )

        self.model = ModelWrapper(model).to(device=get_accelerator().get_current_device())

        if self.verbose:
            logger.info(
                f"After the shard, Rank: [{dist.get_rank()}], model size: {get_model_size(self.model)} GB, model's device is: {model.device}"
            )

        if pretrained_path:
            from colossalai.inference.core.plugin import InferCheckpoint_io

            cpt_io = InferCheckpoint_io()
            if_has_index_file, model_index_file = has_index_file(pretrained_path)
            assert if_has_index_file, "the model path is invalid"
            cpt_io.load_model(self.model, model_index_file)

        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = init_gpu_memory - free_gpu_memory
        if self.verbose:
            logger.info(
                f"Rank [{dist.get_rank()}], Model Weight Max Occupy {peak_memory / (1024 ** 3)} GB, Model size: {get_model_size(self.model)} GB"
            )

    def _shardformer(
        self,
        model: nn.Module,
        model_policy: Policy,
        stage_manager: PipelineStageManager = None,
        tp_group: ProcessGroupMesh = None,
    ) -> nn.Module:
        """
        Initialize ShardConfig and replace the model with shardformer.

        Args:
            model (nn.Module): Path or nn.Module of this model.
            model_policy (Policy): The policy to shardformer model which is determined by the model type.
            stage_manager (PipelineStageManager, optional): Used to manage pipeline stages. Defaults to None.
            tp_group (ProcessGroupMesh, optional): Used to manage the process TP group mesh. Defaults to None.

        Returns:
            nn.Module: The model optimized by Shardformer.
        """

        shardconfig = ShardConfig(
            tensor_parallel_process_group=tp_group,
            pipeline_stage_manager=stage_manager,
            enable_tensor_parallelism=(self.inference_config.tp_size > 1),
            enable_fused_normalization=False,
            enable_all_optimization=False,
            enable_flash_attention=False,
            enable_jit_fused=False,
            enable_sequence_parallelism=False,
        )
        shardformer = ShardFormer(shard_config=shardconfig)
        shard_model, _ = shardformer.optimize(model, model_policy)
        return shard_model

    def exposed_compute_only_for_test(self):
        dist_rank = dist.get_rank()

        # Dummy data for each worker
        data = torch.tensor([dist_rank], dtype=torch.float).cuda(dist_rank)
        dist.barrier()

        # Perform distributed all_reduce
        dist.all_reduce(data, op=dist.ReduceOp.SUM)

        dist.barrier()
        logger.info(f"Worker rank {dist_rank}: Sum after all_reduce: {data.item()}")

        return data.item()
