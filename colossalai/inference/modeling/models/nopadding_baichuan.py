# This code is adapted from huggingface baichuan model: hhttps://huggingface.co/baichuan-inc/Baichuan2-13B-Base/blob/main/modeling_baichuan.py
import itertools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.inference.config import ModelShardInferenceConfig
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.backends.attention_backend import AttentionMetaData, get_attention_backend
from colossalai.inference.modeling.backends.pre_attention_backend import get_pre_attention_backend
from colossalai.inference.modeling.models.nopadding_llama import NopadLlamaMLP
from colossalai.inference.utils import get_alibi_slopes
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import rms_layernorm
from colossalai.logging import get_dist_logger
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.tensor.d_tensor import Layout, distribute_tensor, is_distributed_tensor

inference_ops = InferenceOpsLoader().load()
logger = get_dist_logger(__name__)


def baichuan_rmsnorm_forward(
    self,
    hidden_states: torch.Tensor,
    norm_output: torch.Tensor,
    residual: torch.Tensor = None,
    use_cuda_kernel: bool = True,
):
    # Used to address the issue of inconsistent epsilon variable names in baichuan2 7b and 13b.
    if hasattr(self, "variance_epsilon"):
        eps = self.variance_epsilon
    elif hasattr(self, "epsilon"):
        eps = self.epsilon
    else:
        TypeError(
            "Currently, the variable name for the epsilon of baichuan7B/13B should be 'variance_epsilon' or 'epsilon'."
        )
    if use_cuda_kernel:
        if residual is not None:
            inference_ops.fused_add_rms_layernorm(hidden_states, residual, self.weight.data, eps)
            return hidden_states, residual

        if norm_output is None:
            norm_output = torch.empty_like(hidden_states)
        inference_ops.rms_layernorm(norm_output, hidden_states, self.weight.data, eps)
        return norm_output, hidden_states
    else:
        return rms_layernorm(hidden_states, self.weight.data, eps, norm_output, residual)


class NopadBaichuanAttention(ParallelModule):
    def __init__(
        self,
        config,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj: ParallelModule = None,
        num_heads: int = None,
        hidden_size: int = None,
        model_shard_infer_config: ModelShardInferenceConfig = None,
        process_group: ProcessGroup = None,
        helper_layout: Layout = None,
    ):
        """This layer will replace the BaichuanAttention.

        Args:
            config (BaichuanConfig): Holding the Baichuan model config.
            attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
            attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
            attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
            attn_oproj (Linear1D_Row, optional): The Linear1D_Row o_proj weight. Defaults to None.
        """
        ParallelModule.__init__(self)
        self.o_proj = attn_oproj

        self.config = config
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.process_group = process_group
        qkv_weight_list = [attn_qproj_w.transpose(0, 1), attn_kproj_w.transpose(0, 1), attn_vproj_w.transpose(0, 1)]
        self.qkv_weight = nn.Parameter(torch.stack(qkv_weight_list, dim=0))

        self.helper_layout = helper_layout
        self.use_cuda_kernel = model_shard_infer_config.use_cuda_kernel
        self.attention_backend = get_attention_backend(model_shard_infer_config)
        self.pre_attention_backend = get_pre_attention_backend(model_shard_infer_config)

        self.alibi_slopes = None
        self.use_alibi_attn = False
        # Used for Baichuan13B
        if config.hidden_size == 5120:
            slopes_start = self.process_group.rank() * num_heads
            self.use_alibi_attn = True
            self.alibi_slopes = get_alibi_slopes(config.num_attention_heads, device=attn_qproj_w.device)[
                slopes_start : slopes_start + num_heads
            ].contiguous()
            self.alibi_slopes = nn.Parameter(self.alibi_slopes)

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> "NopadBaichuanAttention":
        """Used for initialize the weight of NopadBaichuanAttention by origin BaichuanAttention.

        Args:
            module (nn.Module): The origin BaichuanAttention layer.
        """

        config = module.config
        q_proj_w, k_proj_w, v_proj_w = module.W_pack.weight.view((module.hidden_size, 3, -1)).transpose(0, 1)

        attn_qproj_w = q_proj_w
        attn_kproj_w = k_proj_w
        attn_vproj_w = v_proj_w
        attn_oproj = module.o_proj
        model_shard_infer_config = kwargs.get("model_shard_infer_config", None)

        helper_layout = (
            module.W_pack.weight.dist_layout
        )  # NOTE this is a hack for the right load/shard of qkv_weight(used in _load_from_state_dict)

        attn_layer = NopadBaichuanAttention(
            config=config,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj=attn_oproj,
            model_shard_infer_config=model_shard_infer_config,
            num_heads=module.num_heads,
            hidden_size=module.hidden_size,
            process_group=process_group,
            helper_layout=helper_layout,
        )

        return attn_layer

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        key = "qkv_weight"
        qkv_w = state_dict[prefix + "W_pack.weight"]

        in_features = qkv_w.size(1)
        out_features = qkv_w.size(0) // 3

        qkv_w.data = qkv_w.view((3, out_features, -1)).transpose(0, 1).reshape(out_features, in_features * 3)

        device_mesh = self.helper_layout.device_mesh
        sharding_spec = self.helper_layout.sharding_spec
        qkv_w = distribute_tensor(qkv_w, device_mesh, sharding_spec)

        qkv_w = qkv_w.transpose(0, 1).reshape(3, in_features, -1)
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
            cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
            high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
        """

        token_nums = hidden_states.size(0)
        # fused qkv
        hidden_states = hidden_states.expand(3, -1, -1)
        query_states, key_states, value_states = (
            torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
        )

        block_size = k_cache.size(-2)

        attn_metadata = AttentionMetaData(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            block_size=block_size,
            kv_seq_len=kv_seq_len,
            sequence_lengths=sequence_lengths,
            sm_scale=sm_scale,
            alibi_slopes=self.alibi_slopes,
            cu_seqlens=cu_seqlens,
            output_tensor=output_tensor,
            use_spec_dec=is_verifier,
            use_alibi_attn=self.use_alibi_attn,
        )

        if is_prompts:  # prefilling stage
            self.pre_attention_backend.prefill(
                attn_metadata,
                cos=cos_sin[0],
                sin=cos_sin[1],
                high_precision=high_precision,
            )
            attn_output = self.attention_backend.prefill(
                attn_metadata,
                token_nums=token_nums,
            )
        else:  # decoding stage
            q_len = tokens_to_verify + 1 if is_verifier else 1

            self.pre_attention_backend.decode(
                attn_metadata,
                cos=cos_sin[0],
                sin=cos_sin[1],
                q_len=q_len,
            )
            attn_output = self.attention_backend.decode(
                attn_metadata,
                fd_inter_tensor=fd_inter_tensor,
                q_len=q_len,
            )

        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def extra_repr(self) -> str:
        return f"qkv_weight_proj MergedLinear1D_Col: in_features={self.qkv_weight.shape[1]}x3, out_features={self.qkv_weight.shape[2]}, bias=False"


# NOTE This will cause difference as out length increases.
class NopadBaichuanMLP(NopadLlamaMLP):
    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        """Used for initialize the weight of NopadBaichuanMLP by origin MLP(Baichuan).

        Args:
            module (nn.Module): The origin MLP(Baichuan) layer.
        """
        mlp_gproj_w = module.gate_proj.weight
        assert is_distributed_tensor(
            module.gate_proj.weight
        ), "gate_proj.weight must be dtensor so we could get the layout of the weight"
        mlp_uproj_w = module.up_proj.weight
        mlp_dproj = module.down_proj

        mlp_layer = NopadBaichuanMLP(
            config=None,
            mlp_gproj_w=mlp_gproj_w,
            mlp_uproj_w=mlp_uproj_w,
            mlp_dproj=mlp_dproj,
            process_group=process_group,
        )

        return mlp_layer
