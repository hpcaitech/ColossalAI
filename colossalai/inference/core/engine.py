import time
from itertools import count
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist
from transformers import AutoConfig, GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from colossalai.accelerator import get_accelerator
from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.batch_bucket import BatchBucket
from colossalai.inference.config import InferenceConfig, InputMetaData
from colossalai.inference.graph_runner import CUDAGraphRunner
from colossalai.inference.modeling.policy import model_policy_map
from colossalai.inference.struct import Sequence
from colossalai.inference.utils import has_index_file
from colossalai.interface import ModelWrapper
from colossalai.logging import get_dist_logger
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.base_policy import Policy

from .request_handler import RequestHandler

__all__ = ["InferenceEngine"]

PP_AXIS, TP_AXIS = 0, 1

_supported_models = {
    "LlamaForCausalLM": LlamaForCausalLM,
}

_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class InferenceEngine:

    """
    InferenceEngine which manages the inference process..

    Args:
        model_or_path (nn.Module or str): Path or nn.Module of this model.
        tokenizer Optional[(Union[PreTrainedTokenizer, PreTrainedTokenizerFast])]: Path of the tokenizer to use.
        inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
        verbose (bool): Determine whether or not to log the generation process.
        model_policy ("Policy"): the policy to shardformer model. It will be determined by the model type if not provided.
    """

    def __init__(
        self,
        model_or_path: Union[nn.Module, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        inference_config: InferenceConfig,
        verbose: bool = False,
        model_policy: Policy = None,
    ) -> None:
        assert inference_config, "Please provide inference_config."
        assert tokenizer, "Please provide a tokenizer, either a defined one or str"
        self.inference_config = inference_config

        self.dtype = inference_config.dtype
        torch.set_default_dtype(self.dtype)
        self.high_precision = inference_config.high_precision

        self.verbose = verbose
        if verbose:
            self.logger = get_dist_logger(__name__)

        # enable memory history, which will
        # add tracebacks and event history to snapshots
        # torch.cuda.memory._record_memory_history()
        self.init_model(model_or_path, model_policy)
        # torch.cuda.memory._dump_snapshot(f"my_snapshot_rank_{dist.get_rank()}.pickle")

        self.generation_config = inference_config.to_generation_config(self.model_config)

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.request_handler = RequestHandler(self.inference_config, self.model_config)
        self.k_cache, self.v_cache = self.request_handler.get_kvcache()
        # DISCUSS maybe move this into batch info?

        self.counter = count()

        self.use_cuda_graph = self.inference_config.use_cuda_graph
        if self.use_cuda_graph:
            self.graph_runners: Dict[int, CUDAGraphRunner] = {}
            self.graph_memory_pool = None  # Set during graph capture.
            if verbose:
                self.logger.info("Colossal AI CUDA Graph Capture on")

            self.capture_model(self.k_cache, self.v_cache)

    def init_model(self, model_or_path: Union[nn.Module, str], model_policy: Policy = None):
        """
        Shard model or/and Load weight

        Args:
            model_or_path Union[nn.Module, str]: path to the checkpoint or model of transformer format.
            model_policy (Policy): the policy to replace the model
        """

        if isinstance(model_or_path, str):
            hf_config = AutoConfig.from_pretrained(model_or_path, trust_remote_code=True)
            arch = getattr(hf_config, "architectures", [])[0]
            model = _supported_models[arch](hf_config)
        else:
            model = model_or_path

        self.model_config = model.config

        torch.cuda.empty_cache()
        init_gpu_memory = torch.cuda.mem_get_info()[0]

        self.device = get_accelerator().get_current_device()
        if self.verbose:
            self.logger.info(f"the device is {self.device}")

        model = model.eval()

        if self.verbose:
            self.logger.info(
                f"Before the shard, Rank: [{dist.get_rank()}], model size: {self.get_model_size(model)} GB, model's device is: {model.device}"
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

        self.model = ModelWrapper(model).to(self.device)

        if self.verbose:
            self.logger.info(
                f"After the shard, Rank: [{dist.get_rank()}], model size: {self.get_model_size(self.model)} GB, model's device is: {model.device}"
            )

        if isinstance(model_or_path, str):
            from colossalai.inference.core.plugin import InferCheckpoint_io

            # from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io import HybridParallelCheckpointIO

            cpt_io = InferCheckpoint_io()
            # cpt_io = HybridParallelCheckpointIO(dp_group=None, pp_group=None, tp_group=pg_mesh.get_group_along_axis(TP_AXIS), zero_stage=0, verbose=True)
            if_has_index_file, model_index_file = has_index_file(model_or_path)
            assert if_has_index_file, "the model path is invalid"
            cpt_io.load_model(self.model, model_index_file)

        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = init_gpu_memory - free_gpu_memory
        if self.verbose:
            self.logger.info(
                f"Rank [{dist.get_rank()}], Model Weight Max Occupy {peak_memory / (1024 ** 3)} GB, Model size: {self.get_model_size(self.model)} GB"
            )

    def get_model_size(self, model: nn.Module):
        """Calculates the total size of the model weights (including biases) in bytes.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            The total size of the model weights in bytes.
        """
        total_size = 0
        for key, param in model.named_parameters():
            total_size += param.element_size() * param.numel()
        return total_size / (1024**3)

    @torch.inference_mode()
    def capture_model(self, k_cache: List[torch.Tensor], v_cache: List[torch.Tensor]):
        assert self.use_cuda_graph, "please turn on the cuda graph"

        if self.verbose:
            self.logger.info("Colossal AI CUDA Graph Capture begin")

        t_capture_begin = time.perf_counter()

        block_size = self.inference_config.block_size
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        max_context_len_to_capture = self.inference_config.max_context_len_to_capture
        max_num_blocks = (max_context_len_to_capture + block_size - 1) // block_size
        input_tokens_ids = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        # self.graph_block_tables = np.zeros((max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)
        self.graph_block_tables = np.full((max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), -1, dtype=np.int32)
        self.graph_block_tables[:, 0] = np.arange(max_num_blocks, max_num_blocks + max(_BATCH_SIZES_TO_CAPTURE))
        self.graph_block_tables[0, :] = np.arange(
            0, max_num_blocks
        )  # NOTE this is a hack to insure cuda grpah could capture the fixed cuda kernel grid in flash decoding, to make the first seqlen as the max_seq_len
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()
        output_tensor = torch.zeros(
            (max_batch_size, self.model_config.num_attention_heads * head_dim), dtype=self.dtype, device=self.device
        )
        fd_inter_tensor = self.request_handler.running_bb.fd_inter_tensor

        max_num_seqs = self.inference_config.max_batch_size
        batch_size_capture_list = [bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= max_num_seqs]
        sequence_lengths = torch.ones(max_batch_size, dtype=torch.int).cuda()
        # NOTE this is a hack to insure cuda grpah could capture the fixed cuda kernel grid in flash decoding, to make the first seqlen as the max_seq_len
        sequence_lengths[0] = torch.tensor(
            self.inference_config.max_context_len_to_capture - 1, dtype=torch.int32
        ).cuda()

        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(batch_size_capture_list):
            if self.verbose:
                self.logger.info(f"batch size {batch_size} graph capturing")

            input_meta_data = InputMetaData(
                block_tables=block_tables[:batch_size],
                sequence_lengths=sequence_lengths[:batch_size],
                fd_inter_tensor=fd_inter_tensor,
                batch_size=batch_size,
                is_prompts=False,
                use_cuda_graph=True,
                high_precision=False,
                kv_seq_len=sequence_lengths[:batch_size].max().item(),
                head_dim=head_dim,
                dtype=self.dtype,
            )

            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(
                input_tokens_ids[:batch_size],
                output_tensor[:batch_size],
                input_meta_data,
                k_caches=k_cache,
                v_caches=v_cache,
                memory_pool=self.graph_memory_pool,
            )
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

        t_capture_end = time.perf_counter()

        if self.verbose:
            self.logger.info(f"CUDA Graph capture time: {t_capture_end - t_capture_begin} s")

    def _verify_config(self) -> None:
        """
        Verify the input config
        """
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"the model type must be nn.Module, but got {type(self.model)}")
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast) and not isinstance(
            self.tokenizer, PreTrainedTokenizer
        ):
            raise TypeError(
                f"the tokenizer type must be PreTrainedTokenizer or PreTrainedTokenizerFast, but got {type(self.tokenizer)}"
            )
        assert (
            self.model.__class__.__name__ in _supported_models.keys()
        ), f"Model {self.model.__class__.__name__} is not supported."

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

    def generate(
        self,
        prompts: List[str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
        request_ids: List[int] = None,
        return_token_ids: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Executing the inference step.

        Args:
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
            request_ids (List[int], optional): The request ID. Defaults to None.
            return_token_ids (bool): Whether to return output token ids. Defaults to False.
            generation_config (GenerationConfig, optional): Huggingface GenerationConfig used for inference. Defaults to None.

        Returns:
            List[str]: Inference result returned by one generation.
        """
        with torch.inference_mode():
            self.generation_config = generation_config
            if prompts is not None or prompts_token_ids is not None:
                self.add_request(request_ids=request_ids, prompts=prompts, prompts_token_ids=prompts_token_ids)

            output_seqs_list = []
            total_tokens_list = []

            # intuition: If user provide a generation config, we should replace the existing one.
            if generation_config is not None:
                self.generation_config = generation_config

            while self.request_handler.check_unfinished_seqs():
                output_seqs_list += self.step()

            output_seqs_list = sorted(output_seqs_list, key=lambda x: int(x.request_id))

            for seq in output_seqs_list:
                total_tokens_list.append(seq.input_token_id + seq.output_token_id)

            output_str = self.tokenizer.batch_decode(total_tokens_list, skip_special_tokens=True)

            if return_token_ids:
                output_tokens_list = [seq.output_token_id for seq in output_seqs_list]
                return output_str, output_tokens_list
            else:
                return output_str

    @property
    def has_prompt_template(self) -> bool:
        """ """
        return self.inference_config.prompt_template is not None

    def format_prompt(self, prompts: Union[List[str], str]) -> Union[List[str], str]:
        """
        This method will format the input prompt according to the prompt template given to the InferenceConfig.
        """
        assert (
            self.has_prompt_template
        ), "Found the prompt_template is None. Please provide a valid prompt_template in InferenceConfig."

        if isinstance(prompts, (list, tuple)):
            return [self.inference_config.prompt_template.format(input_text=prompt) for prompt in prompts]
        elif isinstance(prompts, str):
            return self.inference_config.rompt_template.format(input_text=prompts)
        else:
            raise TypeError(f"Expected the input prompt to be one of list, tuple, or str, but got {type(prompts)}.")

    def add_request(
        self,
        request_ids: List[int] = None,
        prompts: List[str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
    ) -> None:
        """
        Add requests.

        Args:
            request_ids (List[int], optional): The request ID. Defaults to None.
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
        """

        # apply the prompt template to the input prompts
        if self.has_prompt_template and prompts is not None:
            prompts = self.format_prompt(prompts)

        block_size = self.inference_config.block_size

        if prompts is not None and not isinstance(prompts, list):
            prompts = [prompts]

        if prompts_token_ids is None:
            assert prompts, "When the prompts_token_ids is none, the input prompt list must be provided."
            prompts_token_ids = self.tokenizer.batch_encode_plus(prompts, padding=self.inference_config.pad_input)[
                "input_ids"
            ]

        if isinstance(prompts_token_ids, list):
            pass
        elif isinstance(prompts_token_ids, torch.Tensor) or isinstance(prompts_token_ids, np.ndarray):
            prompts_token_ids = prompts_token_ids.tolist()
        else:
            raise TypeError(
                f"The dtype of prompts_token_ids must be one of list, torch.Tensor, np.ndarray, but got {type(prompts_token_ids)}."
            )

        assert (
            len(prompts_token_ids[0]) <= self.inference_config.max_input_len
        ), f"The length of input prompts {len(prompts_token_ids[0])} must be less than max_input_len {self.inference_config.max_input_len}."

        prompts_num = len(prompts_token_ids)

        for i in range(prompts_num):
            if request_ids:
                if not isinstance(request_ids, list):
                    request_ids = [request_ids]
                assert isinstance(
                    request_ids[0], int
                ), f"The request_id type must be int, but got {type(request_ids[0])}"
                assert len(request_ids) == prompts_num
                request_id = request_ids[i]
            else:
                request_id = next(self.counter)
            if prompts == None:
                prompt = None
            else:
                prompt = prompts[i]

            sequence = Sequence(
                request_id,
                prompt,
                prompts_token_ids[i],
                block_size,
                None,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.inference_config.max_output_len,
            )
            self.request_handler.add_sequence(sequence)

    def prepare_input(self, batch: BatchBucket) -> Tuple[torch.Tensor, torch.Tensor, InputMetaData]:
        input_ids = batch.get_1D_inputs()

        sequence_lengths = batch.get_sequence_lengths()
        if batch.is_prompts:
            output_tensor = torch.zeros(
                (sequence_lengths.sum().item(), batch.num_heads * batch.head_dim),
                dtype=batch.dtype,
                device=batch.device,
            )
        else:
            output_tensor = torch.zeros(
                (batch.current_batch_size, batch.num_heads * batch.head_dim), dtype=batch.dtype, device=batch.device
            )

        # only when we have the graph for specific decoding batch size can we use the cuda graph for inference
        use_cuda_graph = False
        if self.use_cuda_graph and not batch.is_prompts and batch.current_batch_size in self.graph_runners.keys():
            use_cuda_graph = True

        input_meta_data = InputMetaData(
            block_tables=batch.get_block_table_tensor(),
            sequence_lengths=sequence_lengths,
            fd_inter_tensor=batch.fd_inter_tensor,
            batch_size=batch.current_batch_size,
            is_prompts=batch.is_prompts,
            use_cuda_kernel=self.inference_config.use_cuda_kernel,
            use_cuda_graph=use_cuda_graph,
            high_precision=self.high_precision,
            kv_seq_len=sequence_lengths.max().item(),
            head_dim=batch.head_dim,
            dtype=batch.dtype,
        )

        return input_ids, output_tensor, input_meta_data

    def step(self) -> List[str]:
        """
        In each step, do the follows:
            1. Run RequestHandler.schedule() and get the batch used for inference.
            2. Get the input, inputinfo and output placeholder from the batchbucket
            3. Run model to generate the next token
            4. Update waiting list and running list in RequestHandler and get finished sequences.
            5. Decode and return finished sequences.

        Returns:
            List[str]: Decoded finished sequences generated by one step.
        """

        batch = self.request_handler.schedule()

        input_token_ids, output_tensor, input_meta_data = self.prepare_input(batch)

        if input_meta_data.use_cuda_graph:
            model_executable = self.graph_runners[input_meta_data.batch_size]
        else:
            model_executable = self.model

        # TODO: padding_id is used for generating attn_mask and will be removed if nopad version is supported.
        logits = model_executable(input_token_ids, output_tensor, input_meta_data, self.k_cache, self.v_cache)
        if self.inference_config.pad_input:
            logits = logits[:, -1, :]
        self.request_handler.search_tokens(self.generation_config, logits)

        finished_sequences = self.request_handler.update()

        return finished_sequences
