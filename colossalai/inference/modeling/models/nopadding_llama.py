# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
import itertools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
)

from colossalai.inference.config import InputMetaData
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import (
    context_attention_unpadded,
    copy_k_to_blocked_cache,
    decoding_fused_rotary_embedding,
    flash_decoding_attention,
    get_xine_cache,
    rms_layernorm,
    rotary_embedding,
)
from colossalai.logging import get_dist_logger
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.tensor.d_tensor import distribute_tensor, is_distributed_tensor

inference_ops = InferenceOpsLoader().load()

logger = get_dist_logger(__name__)

try:
    from flash_attn import flash_attn_varlen_func

    use_flash_attn2 = True
except ImportError:
    use_flash_attn2 = False
    logger.warning(f"flash_attn2 has not been installed yet, we will use triton flash attn instead.")


def llama_causal_lm_forward(
    self: LlamaForCausalLM,
    input_tokens_ids: torch.Tensor,
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
) -> torch.Tensor:
    """This function will replace the forward function of LlamaForCausalLM.

    Args:
        batch (BatchInfo): It stores the necessary input information for this inference.
        k_caches (List[torch.Tensor]): It holds the GPU memory for the key cache.
        v_caches (List[torch.Tensor]): It holds the GPU memory for the value cache.
        high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
    """

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    hidden_states = llama_model_forward(
        self.model,
        input_tokens_ids=input_tokens_ids,
        output_tensor=output_tensor,
        inputmetadata=inputmetadata,
        k_caches=k_caches,
        v_caches=v_caches,
        use_cuda_kernel=inputmetadata.use_cuda_kernel,  # Note currently the cuda kernel of layernorm, rotary_embedding_and_cache_copy couldn't pass the unitest but triton kernel could
        high_precision=inputmetadata.high_precision,
    )

    logits = self.lm_head(hidden_states)
    return logits


def llama_model_forward(
    self: LlamaModel,
    input_tokens_ids: torch.Tensor,
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
    use_cuda_kernel: Optional[bool] = True,
    high_precision: bool = False,
) -> torch.Tensor:
    """This function will replace the forward function of LlamaModel.

    Args:
        batch (BatchInfo, optional): It stores the necessary input information for this inference.. Defaults to None.
        k_caches (List[torch.Tensor], optional): It holds the GPU memory for the key cache. Defaults to None.
        v_caches (List[torch.Tensor], optional): It holds the GPU memory for the value cache. Defaults to None.
        high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
    """
    block_tables = inputmetadata.block_tables
    sequence_lengths = inputmetadata.sequence_lengths
    kv_seq_len = inputmetadata.kv_seq_len

    # NOTE (yuanheng-zhao): fow now, only triton kernels support verification process
    # during speculative-decoding (`q_len > 1`)
    # We will expicitly disable `use_cuda_kernel` here when speculative-decoding is enabled
    if inputmetadata.use_spec_dec and use_cuda_kernel:
        use_cuda_kernel = False
        logger.warning("CUDA kernel is disabled for speculative-decoding.")

    hidden_states = self.embed_tokens(input_tokens_ids)

    cu_seqlens = None

    # NOTE (yuanheng-zhao): we do not use cuda kernels for speculative-decoding for now
    if inputmetadata.use_spec_dec:
        # For speculative-decoding Prefill and Verifying Stage
        if inputmetadata.is_prompts:
            # output tensor shape is the same as normal Prefill Stage
            rotary_indexes = [torch.arange(0, length) for length in sequence_lengths]
        else:
            # the number of tokens to be verified in parallel plus the correct token in the last step
            n_tokens = inputmetadata.num_tokens_to_verify + 1
            assert n_tokens == hidden_states.size(0)
            rotary_indexes = [(length - n_tokens + i).view(-1) for i in range(n_tokens) for length in sequence_lengths]
        rotary_indexes = torch.cat(rotary_indexes, dim=-1)
        cos_sin = (self._cos_cached[rotary_indexes], self._sin_cached[rotary_indexes])

    elif use_cuda_kernel:
        if inputmetadata.dtype != torch.float32 and use_flash_attn2:
            cu_seqlens = F.pad(torch.cumsum(sequence_lengths, dim=0, dtype=torch.torch.int32), (1, 0))

        hidden_dim = self._cos_cached.size(-1)
        total_length = hidden_states.size(0)
        cos = torch.empty((total_length, hidden_dim), dtype=self._cos_cached.dtype, device=self._cos_cached.device)
        sin = torch.empty((total_length, hidden_dim), dtype=self._sin_cached.dtype, device=self._sin_cached.device)
        inference_ops.get_cos_and_sin(
            self._cos_cached, self._sin_cached, cos, sin, sequence_lengths, kv_seq_len, inputmetadata.is_prompts
        )
        cos_sin = (cos, sin)
    else:
        cos_sin = get_xine_cache(sequence_lengths, self._cos_cached, self._sin_cached, inputmetadata.is_prompts)

    sm_scale = 1.0 / (inputmetadata.head_dim**0.5)

    norm_output = torch.empty_like(hidden_states)
    tokens_to_verify = inputmetadata.num_tokens_to_verify if inputmetadata.use_spec_dec else None
    residual = None

    for layer_id, decoder_layer in enumerate(self.layers):
        hidden_states, residual = decoder_layer(
            hidden_states,
            residual=residual,
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            is_prompts=inputmetadata.is_prompts,
            is_verifier=inputmetadata.use_spec_dec,
            tokens_to_verify=tokens_to_verify,
            sequence_lengths=sequence_lengths,
            cos_sin=cos_sin,
            fd_inter_tensor=inputmetadata.fd_inter_tensor,
            kv_seq_len=kv_seq_len,
            output_tensor=output_tensor,
            norm_output=norm_output,
            sm_scale=sm_scale,
            use_cuda_kernel=use_cuda_kernel,
            cu_seqlens=cu_seqlens,
            high_precision=high_precision,
        )

    if inputmetadata.is_prompts:
        seq_len_cumsum = sequence_lengths.cumsum(dim=0)
        hidden_states = hidden_states[seq_len_cumsum - 1].contiguous()
        residual = residual[seq_len_cumsum - 1].contiguous()
        norm_output = torch.empty_like(hidden_states)
    hidden_states, _ = self.norm(hidden_states, norm_output, residual, use_cuda_kernel)

    return hidden_states


def llama_decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    block_tables: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    cos_sin: Tuple[torch.Tensor],
    fd_inter_tensor: FDIntermTensors,
    is_prompts: bool = True,
    is_verifier: bool = False,
    tokens_to_verify: int = None,
    kv_seq_len: int = 0,
    output_tensor: torch.Tensor = None,
    norm_output: torch.Tensor = None,
    sm_scale: int = None,
    use_cuda_kernel: bool = True,
    cu_seqlens: torch.Tensor = None,
    high_precision: bool = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """This function will replace the forward function of LlamaDecoderLayer.

    Args:
        hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        residual (torch.Tensor): shape [token_num, embed_dim], used to be added to hidden_states in out_proj.
        block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
            storing mapping of token_position_id -> block_id.
        k_cache (torch.Tensor): It holds the GPU memory for the key cache.
        v_cache (torch.Tensor): It holds the GPU memory for the key cache.
        sequence_lengths (torch.Tensor): Holding the sequence length of each sequence.
        cos_sin (Tuple[torch.Tensor]): Holding cos and sin.
        fd_inter_tensor (FDIntermTensors): Holding tensors used for
            storing intermediate values in flash-decoding.
        is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
        kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
        output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
        norm_output (torch.Tensor, optional): The mid tensor holds the output of layernorm. Defaults to None.
        sm_scale (int, optional): Used for flash attention. Defaults to None.
        use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
        cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
        high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
    """

    hidden_states, residual = self.input_layernorm(hidden_states, norm_output, residual, use_cuda_kernel)
    # Self Attention
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        is_verifier=is_verifier,
        tokens_to_verify=tokens_to_verify,
        sequence_lengths=sequence_lengths,
        cos_sin=cos_sin,
        fd_inter_tensor=fd_inter_tensor,
        kv_seq_len=kv_seq_len,
        output_tensor=output_tensor,
        sm_scale=sm_scale,
        use_cuda_kernel=use_cuda_kernel,
        cu_seqlens=cu_seqlens,
        high_precision=high_precision,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, norm_output, residual, use_cuda_kernel)
    hidden_states = self.mlp(hidden_states)

    return hidden_states, residual


def llama_rmsnorm_forward(
    self: LlamaRMSNorm,
    hidden_states: torch.Tensor,
    norm_output: torch.Tensor,
    residual: torch.Tensor = None,
    use_cuda_kernel: bool = True,
):
    if use_cuda_kernel:
        if residual is not None:
            inference_ops.fused_add_rms_layernorm(hidden_states, residual, self.weight.data, self.variance_epsilon)
            return hidden_states, residual

        if norm_output is None:
            norm_output = torch.empty_like(hidden_states)
        inference_ops.rms_layernorm(norm_output, hidden_states, self.weight.data, self.variance_epsilon)
        return norm_output, hidden_states
    else:
        return rms_layernorm(hidden_states, self.weight.data, self.variance_epsilon, norm_output, residual)


class NopadLlamaMLP(LlamaMLP, ParallelModule):
    def __init__(
        self,
        config: LlamaConfig,
        mlp_gproj_w: torch.Tensor = None,
        mlp_uproj_w: torch.Tensor = None,
        mlp_dproj: ParallelModule = None,
        process_group: ProcessGroup = None,
    ):
        """A Unified Layer for

        Args:
            config (LlamaConfig): Holding the Llama model config.
            mlp_gproj_w (torch.Tensor, optional): The transposed gate_proj weight. Defaults to None.
            mlp_uproj_w (torch.Tensor, optional): The transposed up_proj weight. Defaults to None.
            mlp_dproj (Linear1D_Row, optional): The Linear1D_Row mlp_dproj weight. Defaults to None.
        """
        ParallelModule.__init__(self)
        self.config = config
        assert is_distributed_tensor(
            mlp_gproj_w
        ), "mlp_gproj_w must be dtensor so we could get the layout of the weight"
        self.helper_layout = (
            mlp_gproj_w.dist_layout
        )  # NOTE this is a hack for the right load/shard of gate_up_weight(used in _load_from_state_dict)
        self.gate_up_weight = nn.Parameter(
            torch.stack([mlp_gproj_w.transpose(0, 1), mlp_uproj_w.transpose(0, 1)], dim=0)
        )
        self.down_proj = mlp_dproj
        self.process_group = process_group

    @staticmethod
    def from_native_module(
        module: LlamaMLP, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        """Used for initialize the weight of NopadLlamaMLP by origin LlamaMLP.

        Args:
            module (LlamaMLP): The origin LlamaMLP layer.
        """

        config = module.config

        mlp_gproj_w = module.gate_proj.weight
        assert is_distributed_tensor(
            module.gate_proj.weight
        ), "gate_proj.weight must be dtensor so we could get the layout of the weight"
        mlp_uproj_w = module.up_proj.weight
        mlp_dproj = module.down_proj

        mlp_layer = NopadLlamaMLP(
            config=config,
            mlp_gproj_w=mlp_gproj_w,
            mlp_uproj_w=mlp_uproj_w,
            mlp_dproj=mlp_dproj,
            process_group=process_group,
        )

        return mlp_layer

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # NOTE This is a hack to ensure we could load the right weight from LlamaMLP checkpoint due to the use of torch.stack(gate_weight, up_weight)

        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        key = "gate_up_weight"
        k1 = "gate_proj.weight"
        k2 = "up_proj.weight"

        gate_w = state_dict[prefix + k1]
        up_w = state_dict[prefix + k2]

        device_mesh = self.helper_layout.device_mesh
        sharding_spec = self.helper_layout.sharding_spec
        gate_w = distribute_tensor(gate_w, device_mesh, sharding_spec)
        up_w = distribute_tensor(up_w, device_mesh, sharding_spec)

        gate_up_w = torch.stack([gate_w.T, up_w.T], dim=0)

        input_param = nn.Parameter(
            gate_up_w
        )  # NOTE gate_up_weight doesn't have to be a distensor, Like input_param = sharded_tensor_to_param(input_param)
        param = local_state[key]

        try:
            with torch.no_grad():
                param.copy_(input_param)
        except Exception as ex:
            error_msgs.append(
                'While copying the parameter named "{}", '
                "whose dimensions in the model are {} and "
                "whose dimensions in the checkpoint are {}, "
                "an exception occurred : {}.".format(key, param.size(), input_param.size(), ex.args)
            )

        strict = False  # to avoid unexpected_keys
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        """
        hidden_states = hidden_states.expand(2, -1, -1)
        gate_up_proj_out = torch.bmm(hidden_states, self.gate_up_weight)
        act_out = inference_ops.silu_and_mul(gate_up_proj_out)

        return self.down_proj(act_out)

    def extra_repr(self) -> str:
        return f"gate_up_proj MergedLinear1D_Col: in_features={self.gate_up_weight.shape[1]}x2, out_features={self.gate_up_weight.shape[2]}, bias=False"


class NopadLlamaAttention(LlamaAttention, ParallelModule):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj: ParallelModule = None,
        process_group: ProcessGroup = None,
        num_heads: int = None,
        hidden_size: int = None,
        num_key_value_heads: int = None,
    ):
        """This layer will replace the LlamaAttention.

        Args:
            config (LlamaConfig): Holding the Llama model config.
            layer_idx (Optional[int], optional): The decode layer id of this attention layer. Defaults to None.
            attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
            attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
            attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
            attn_oproj (Linear1D_Row, optional): The Linear1D_Row o_proj weight. Defaults to None.
        """
        ParallelModule.__init__(self)
        self.config = config
        self.layer_idx = layer_idx

        self.o_proj = attn_oproj
        self.process_group = process_group

        self.attention_dropout = config.attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.num_heads == self.num_key_value_heads:
            qkv_weight_list = [attn_qproj_w.transpose(0, 1), attn_kproj_w.transpose(0, 1), attn_vproj_w.transpose(0, 1)]
            self.qkv_weight = nn.Parameter(torch.stack(qkv_weight_list, dim=0))
            self.helper_layout = (
                attn_qproj_w.dist_layout
            )  # NOTE this is a hack for the right load/shard of qkv_weight(used in _load_from_state_dict)
        else:
            self.q_proj_weight = nn.Parameter(attn_qproj_w.transpose(0, 1).contiguous())
            self.k_proj_weight = nn.Parameter(attn_kproj_w.transpose(0, 1).contiguous())
            self.v_proj_weight = nn.Parameter(attn_vproj_w.transpose(0, 1).contiguous())

    @staticmethod
    def from_native_module(
        module: LlamaAttention, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        """Used for initialize the weight of NopadLlamaAttention by origin LlamaAttention.

        Args:
            module (LlamaAttention): The origin LlamaAttention layer.
        """

        config = module.config
        layer_idx = module.layer_idx

        attn_qproj_w = module.q_proj.weight
        attn_kproj_w = module.k_proj.weight
        attn_vproj_w = module.v_proj.weight
        assert is_distributed_tensor(attn_qproj_w), "attn_qproj_w must be dist tensor"
        attn_oproj = module.o_proj

        attn_layer = NopadLlamaAttention(
            config=config,
            layer_idx=layer_idx,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj=attn_oproj,
            process_group=process_group,
            num_heads=module.num_heads,
            hidden_size=module.hidden_size,
            num_key_value_heads=module.num_key_value_heads,
        )

        return attn_layer

    # Replace transformers.models.llama.modeling_llama.LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_tables: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sequence_lengths: torch.Tensor,
        cos_sin: Tuple[torch.Tensor],
        fd_inter_tensor: FDIntermTensors,
        is_prompts: bool = True,
        is_verifier: bool = False,
        tokens_to_verify: int = None,
        kv_seq_len: int = 0,
        output_tensor: torch.Tensor = None,
        sm_scale: int = None,
        use_cuda_kernel: bool = True,
        cu_seqlens: torch.Tensor = None,
        high_precision: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
            block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
                storing mapping of token_position_id -> block_id.
            k_cache (torch.Tensor): It holds the GPU memory for the key cache.
            v_cache (torch.Tensor): It holds the GPU memory for the key cache.
            sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence.
            cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin.
            fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
                storing intermediate values in flash-decoding.
            is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
            kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
            output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
            sm_scale (int, optional): Used for flash attention. Defaults to None.
            use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
            cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
            high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
        """

        token_nums = hidden_states.size(0)

        if self.num_heads != self.num_key_value_heads:
            query_states = torch.mm(hidden_states, self.q_proj_weight).view(-1, self.num_heads, self.head_dim)
            key_states = torch.mm(hidden_states, self.k_proj_weight).view(-1, self.num_key_value_heads, self.head_dim)
            value_states = torch.mm(hidden_states, self.v_proj_weight).view(-1, self.num_key_value_heads, self.head_dim)
        else:
            # fused qkv
            hidden_states = hidden_states.expand(3, -1, -1)
            query_states, key_states, value_states = (
                torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
            )

        block_size = k_cache.size(-2)

        if is_prompts:
            if not is_verifier and use_cuda_kernel and query_states.dtype != torch.float32 and use_flash_attn2:
                # flash attn 2 currently only supports FP16/BF16.
                inference_ops.rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1], high_precision)
                inference_ops.context_kv_cache_memcpy(
                    key_states, value_states, k_cache, v_cache, sequence_lengths, cu_seqlens, block_tables, kv_seq_len
                )

                attn_output = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=kv_seq_len,
                    max_seqlen_k=kv_seq_len,
                    dropout_p=0.0,
                    softmax_scale=sm_scale,
                    causal=True,
                )
                attn_output = attn_output.view(token_nums, -1)
            else:
                rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])
                attn_output = context_attention_unpadded(
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    context_lengths=sequence_lengths,
                    block_tables=block_tables,
                    block_size=block_size,
                    output=output_tensor,
                    max_seq_len=kv_seq_len,
                    sm_scale=sm_scale,
                    use_new_kcache_layout=use_cuda_kernel,
                )
        else:
            q_len = tokens_to_verify + 1 if is_verifier else 1

            if use_cuda_kernel:
                inference_ops.rotary_embedding_and_cache_copy(
                    query_states,
                    key_states,
                    value_states,
                    cos_sin[0],
                    cos_sin[1],
                    k_cache,
                    v_cache,
                    sequence_lengths,
                    block_tables,
                    high_precision,
                )
                inference_ops.flash_decoding_attention(
                    output_tensor,
                    query_states,
                    k_cache,
                    v_cache,
                    sequence_lengths,
                    block_tables,
                    block_size,
                    kv_seq_len,
                    fd_inter_tensor.mid_output,
                    fd_inter_tensor.exp_sums,
                    fd_inter_tensor.max_logits,
                    None,
                    sm_scale,
                )
                attn_output = output_tensor
            else:
                if is_verifier:
                    rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])
                    copy_k_to_blocked_cache(
                        key_states, k_cache, kv_lengths=sequence_lengths, block_tables=block_tables, n=q_len
                    )
                    copy_k_to_blocked_cache(
                        value_states, v_cache, kv_lengths=sequence_lengths, block_tables=block_tables, n=q_len
                    )
                else:
                    decoding_fused_rotary_embedding(
                        query_states,
                        key_states,
                        value_states,
                        cos_sin[0],
                        cos_sin[1],
                        k_cache,
                        v_cache,
                        block_tables,
                        sequence_lengths,
                    )
                attn_output = flash_decoding_attention(
                    q=query_states,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    kv_seq_len=sequence_lengths,
                    block_tables=block_tables,
                    block_size=block_size,
                    max_seq_len_in_batch=kv_seq_len,
                    output=output_tensor,
                    mid_output=fd_inter_tensor.mid_output,
                    mid_output_lse=fd_inter_tensor.mid_output_lse,
                    sm_scale=sm_scale,
                    kv_group_num=self.num_key_value_groups,
                    q_len=q_len,
                )

        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # NOTE This is a hack to ensure we could load the right weight from LlamaAttention checkpoint due to the use of torch.stack(q_weight, k_weight, v_weight)
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        key = "qkv_weight"
        k1 = "q_proj.weight"
        k2 = "k_proj.weight"
        k3 = "v_proj.weight"
        q_w = state_dict[prefix + k1]
        k_w = state_dict[prefix + k2]
        v_w = state_dict[prefix + k3]

        device_mesh = self.helper_layout.device_mesh
        sharding_spec = self.helper_layout.sharding_spec
        q_w = distribute_tensor(q_w, device_mesh, sharding_spec)
        k_w = distribute_tensor(k_w, device_mesh, sharding_spec)
        v_w = distribute_tensor(v_w, device_mesh, sharding_spec)

        qkv_w = torch.stack([q_w.T, k_w.T, v_w.T], dim=0)

        input_param = nn.Parameter(
            qkv_w
        )  # NOTE qkv_weight doesn't have to be a distensor, Like input_param = sharded_tensor_to_param(input_param)

        param = local_state[key]

        try:
            with torch.no_grad():
                param.copy_(input_param)
        except Exception as ex:
            error_msgs.append(
                'While copying the parameter named "{}", '
                "whose dimensions in the model are {} and "
                "whose dimensions in the checkpoint are {}, "
                "an exception occurred : {}.".format(key, param.size(), input_param.size(), ex.args)
            )

        strict = False  # to avoid unexpected_keys
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def extra_repr(self) -> str:
        return f"qkv_weight_proj MergedLinear1D_Col: in_features={self.qkv_weight.shape[1]}x3, out_features={self.qkv_weight.shape[2]}, bias=False"
