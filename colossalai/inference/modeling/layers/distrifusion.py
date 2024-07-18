# Code refer and adapted from:
# https://github.com/huggingface/diffusers/blob/v0.29.0-release/src/diffusers
# https://github.com/PipeFusion/PipeFusion

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models import attention_processor
from diffusers.models.attention import Attention
from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from torch import nn
from torch.distributed import ProcessGroup

from colossalai.inference.config import ModelShardInferenceConfig
from colossalai.logging import get_dist_logger
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.utils import get_current_device

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


logger = get_dist_logger(__name__)


# adapted from https://github.com/huggingface/diffusers/blob/v0.29.0-release/src/diffusers/models/transformers/transformer_2d.py
def PixArtAlphaTransformer2DModel_forward(
    self: PixArtTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Dict[str, torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    assert hasattr(
        self, "patched_parallel_size"
    ), "please check your policy, `Transformer2DModel` Must have attribute `patched_parallel_size`"

    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
    #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
    #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None and attention_mask.ndim == 2:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 1. Input
    batch_size = hidden_states.shape[0]
    height, width = (
        hidden_states.shape[-2] // self.config.patch_size,
        hidden_states.shape[-1] // self.config.patch_size,
    )
    hidden_states = self.pos_embed(hidden_states)

    timestep, embedded_timestep = self.adaln_single(
        timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
    )

    if self.caption_projection is not None:
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

    # 2. Blocks
    for block in self.transformer_blocks:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
        )

    # 3. Output
    shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)).chunk(
        2, dim=1
    )
    hidden_states = self.norm_out(hidden_states)
    # Modulation
    hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.squeeze(1)

    # unpatchify
    hidden_states = hidden_states.reshape(
        shape=(
            -1,
            height // self.patched_parallel_size,
            width,
            self.config.patch_size,
            self.config.patch_size,
            self.out_channels,
        )
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(
            -1,
            self.out_channels,
            height // self.patched_parallel_size * self.config.patch_size,
            width * self.config.patch_size,
        )
    )

    # enable Distrifusion Optimization
    if hasattr(self, "patched_parallel_size"):
        from torch import distributed as dist

        if (getattr(self, "output_buffer", None) is None) or (self.output_buffer.shape != output.shape):
            self.output_buffer = torch.empty_like(output)
        if (getattr(self, "buffer_list", None) is None) or (self.buffer_list[0].shape != output.shape):
            self.buffer_list = [torch.empty_like(output) for _ in range(self.patched_parallel_size)]
        output = output.contiguous()
        dist.all_gather(self.buffer_list, output, async_op=False)
        torch.cat(self.buffer_list, dim=2, out=self.output_buffer)
        output = self.output_buffer

    return (output,)


# adapted from https://github.com/huggingface/diffusers/blob/v0.29.0-release/src/diffusers/models/transformers/transformer_sd3.py
def SD3Transformer2DModel_forward(
    self: SD3Transformer2DModel,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.FloatTensor]:

    assert hasattr(
        self, "patched_parallel_size"
    ), "please check your policy, `Transformer2DModel` Must have attribute `patched_parallel_size`"

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    for block in self.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
        )

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # unpatchify
    patch_size = self.config.patch_size
    height = height // patch_size // self.patched_parallel_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )

    # enable Distrifusion Optimization
    if hasattr(self, "patched_parallel_size"):
        from torch import distributed as dist

        if (getattr(self, "output_buffer", None) is None) or (self.output_buffer.shape != output.shape):
            self.output_buffer = torch.empty_like(output)
        if (getattr(self, "buffer_list", None) is None) or (self.buffer_list[0].shape != output.shape):
            self.buffer_list = [torch.empty_like(output) for _ in range(self.patched_parallel_size)]
        output = output.contiguous()
        dist.all_gather(self.buffer_list, output, async_op=False)
        torch.cat(self.buffer_list, dim=2, out=self.output_buffer)
        output = self.output_buffer

    return (output,)


# Code adapted from: https://github.com/PipeFusion/PipeFusion/blob/main/pipefuser/modules/dit/patch_parallel/patchembed.py
class DistrifusionPatchEmbed(ParallelModule):
    def __init__(
        self,
        module: PatchEmbed,
        process_group: Union[ProcessGroup, List[ProcessGroup]],
        model_shard_infer_config: ModelShardInferenceConfig = None,
    ):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank(group=process_group)
        self.patched_parallelism_size = model_shard_infer_config.patched_parallelism_size

    @staticmethod
    def from_native_module(module: PatchEmbed, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs):
        model_shard_infer_config = kwargs.get("model_shard_infer_config", None)
        distrifusion_embed = DistrifusionPatchEmbed(
            module, process_group, model_shard_infer_config=model_shard_infer_config
        )
        return distrifusion_embed

    def forward(self, latent):
        module = self.module
        if module.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // module.patch_size, latent.shape[-1] // module.patch_size

        latent = module.proj(latent)
        if module.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if module.layer_norm:
            latent = module.norm(latent)
        if module.pos_embed is None:
            return latent.to(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if module.pos_embed_max_size:
            pos_embed = module.cropped_pos_embed(height, width)
        else:
            if module.height != height or module.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=module.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=module.base_size,
                    interpolation_scale=module.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = module.pos_embed

        b, c, h = pos_embed.shape
        pos_embed = pos_embed.view(b, self.patched_parallelism_size, -1, h)[:, self.rank]

        return (latent + pos_embed).to(latent.dtype)


# Code adapted from: https://github.com/PipeFusion/PipeFusion/blob/main/pipefuser/modules/dit/patch_parallel/conv2d.py
class DistrifusionConv2D(ParallelModule):

    def __init__(
        self,
        module: nn.Conv2d,
        process_group: Union[ProcessGroup, List[ProcessGroup]],
        model_shard_infer_config: ModelShardInferenceConfig = None,
    ):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank(group=process_group)
        self.patched_parallelism_size = model_shard_infer_config.patched_parallelism_size

    @staticmethod
    def from_native_module(module: nn.Conv2d, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs):
        model_shard_infer_config = kwargs.get("model_shard_infer_config", None)
        distrifusion_conv = DistrifusionConv2D(module, process_group, model_shard_infer_config=model_shard_infer_config)
        return distrifusion_conv

    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, h, w = x.shape

        stride = self.module.stride[0]
        padding = self.module.padding[0]

        output_h = x.shape[2] // stride // self.patched_parallelism_size
        idx = dist.get_rank()
        h_begin = output_h * idx * stride - padding
        h_end = output_h * (idx + 1) * stride + padding
        final_padding = [padding, padding, 0, 0]
        if h_begin < 0:
            h_begin = 0
            final_padding[2] = padding
        if h_end > h:
            h_end = h
            final_padding[3] = padding
        sliced_input = x[:, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, final_padding, mode="constant")
        return F.conv2d(
            padded_input,
            self.module.weight,
            self.module.bias,
            stride=stride,
            padding="valid",
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.sliced_forward(input)
        return output


# Code adapted from: https://github.com/huggingface/diffusers/blob/v0.29.0-release/src/diffusers/models/attention_processor.py
class DistrifusionFusedAttention(ParallelModule):

    def __init__(
        self,
        module: attention_processor.Attention,
        process_group: Union[ProcessGroup, List[ProcessGroup]],
        model_shard_infer_config: ModelShardInferenceConfig = None,
    ):
        super().__init__()
        self.counter = 0
        self.module = module
        self.buffer_list = None
        self.kv_buffer_idx = dist.get_rank(group=process_group)
        self.patched_parallelism_size = model_shard_infer_config.patched_parallelism_size
        self.handle = None
        self.process_group = process_group
        self.warm_step = 5  # for warmup

    @staticmethod
    def from_native_module(
        module: attention_processor.Attention, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        model_shard_infer_config = kwargs.get("model_shard_infer_config", None)
        return DistrifusionFusedAttention(
            module=module,
            process_group=process_group,
            model_shard_infer_config=model_shard_infer_config,
        )

    def _forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        kv = torch.cat([key, value], dim=-1)  # shape of kv now: (bs, seq_len // parallel_size, dim * 2)

        if self.patched_parallelism_size == 1:
            full_kv = kv
        else:
            if self.buffer_list is None:  # buffer not created
                full_kv = torch.cat([kv for _ in range(self.patched_parallelism_size)], dim=1)
            elif self.counter <= self.warm_step:
                # logger.info(f"warmup: {self.counter}")
                dist.all_gather(
                    self.buffer_list,
                    kv,
                    group=self.process_group,
                    async_op=False,
                )
                full_kv = torch.cat(self.buffer_list, dim=1)
            else:
                # logger.info(f"use old kv to infer: {self.counter}")
                self.buffer_list[self.kv_buffer_idx].copy_(kv)
                full_kv = torch.cat(self.buffer_list, dim=1)
                assert self.handle is None, "we should maintain the kv of last step"
                self.handle = dist.all_gather(self.buffer_list, kv, group=self.process_group, async_op=True)

        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )  # NOTE(@lry89757) for torch >= 2.2, flash attn has been already integrated into scaled_dot_product_attention, https://pytorch.org/blog/pytorch2-2/
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:

        if self.handle is not None:
            self.handle.wait()
            self.handle = None

        b, l, c = hidden_states.shape
        kv_shape = (b, l, self.module.to_k.out_features * 2)
        if self.patched_parallelism_size > 1 and (self.buffer_list is None or self.buffer_list[0].shape != kv_shape):

            self.buffer_list = [
                torch.empty(kv_shape, dtype=hidden_states.dtype, device=get_current_device())
                for _ in range(self.patched_parallelism_size)
            ]

            self.counter = 0

        attn_parameters = set(inspect.signature(self.module.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.module.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        output = self._forward(
            self.module,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        self.counter += 1

        return output


# Code adapted from: https://github.com/PipeFusion/PipeFusion/blob/main/pipefuser/modules/dit/patch_parallel/attn.py
class DistriSelfAttention(ParallelModule):
    def __init__(
        self,
        module: Attention,
        process_group: Union[ProcessGroup, List[ProcessGroup]],
        model_shard_infer_config: ModelShardInferenceConfig = None,
    ):
        super().__init__()
        self.counter = 0
        self.module = module
        self.buffer_list = None
        self.kv_buffer_idx = dist.get_rank(group=process_group)
        self.patched_parallelism_size = model_shard_infer_config.patched_parallelism_size
        self.handle = None
        self.process_group = process_group
        self.warm_step = 3  # for warmup

    @staticmethod
    def from_native_module(
        module: Attention, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        model_shard_infer_config = kwargs.get("model_shard_infer_config", None)
        return DistriSelfAttention(
            module=module,
            process_group=process_group,
            model_shard_infer_config=model_shard_infer_config,
        )

    def _forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0):
        attn = self.module
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)

        encoder_hidden_states = hidden_states
        k = self.module.to_k(encoder_hidden_states)
        v = self.module.to_v(encoder_hidden_states)
        kv = torch.cat([k, v], dim=-1)  # shape of kv now: (bs, seq_len // parallel_size, dim * 2)

        if self.patched_parallelism_size == 1:
            full_kv = kv
        else:
            if self.buffer_list is None:  # buffer not created
                full_kv = torch.cat([kv for _ in range(self.patched_parallelism_size)], dim=1)
            elif self.counter <= self.warm_step:
                # logger.info(f"warmup: {self.counter}")
                dist.all_gather(
                    self.buffer_list,
                    kv,
                    group=self.process_group,
                    async_op=False,
                )
                full_kv = torch.cat(self.buffer_list, dim=1)
            else:
                # logger.info(f"use old kv to infer: {self.counter}")
                self.buffer_list[self.kv_buffer_idx].copy_(kv)
                full_kv = torch.cat(self.buffer_list, dim=1)
                assert self.handle is None, "we should maintain the kv of last step"
                self.handle = dist.all_gather(self.buffer_list, kv, group=self.process_group, async_op=True)

        if HAS_FLASH_ATTN:
            # flash attn
            key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)

            hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        else:
            # naive attn
            key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        # async preallocates memo buffer
        if self.handle is not None:
            self.handle.wait()
            self.handle = None

        b, l, c = hidden_states.shape
        kv_shape = (b, l, self.module.to_k.out_features * 2)
        if self.patched_parallelism_size > 1 and (self.buffer_list is None or self.buffer_list[0].shape != kv_shape):

            self.buffer_list = [
                torch.empty(kv_shape, dtype=hidden_states.dtype, device=get_current_device())
                for _ in range(self.patched_parallelism_size)
            ]

            self.counter = 0

        output = self._forward(hidden_states, scale=scale)

        self.counter += 1
        return output
