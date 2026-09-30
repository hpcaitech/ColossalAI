# Modified from colossalai/shardformer/modeling/mixtral.py and colossalai/shardformer/modeling/qwen3.py
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
    load_balancing_loss_func,
)
from transformers.utils import logging

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import (
    DPGradScalerIn,
    DPGradScalerOut,
    EPGradScalerIn,
    EPGradScalerOut,
    all_to_all_uneven,
)
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.quantization.fp8 import all_reduce_fp8
from colossalai.shardformer.layer._operation import gather_sp_output, split_forward_gather_backward
from colossalai.shardformer.layer.linear import Linear1D_Col, Linear1D_Row, ParallelModule
from colossalai.shardformer.layer.utils import is_share_sp_tp
from colossalai.shardformer.shard import ShardConfig
from colossalai.shardformer.shard.utils import set_tensors_to_none
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_group

from ..layer import ColoAttention


class EPQwen3MoeSparseMoeBlock(ParallelModule):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(f"Please use `from_native_module` to create an instance of {self.__class__.__name__}")

    def setup_process_groups(
        self,
        tp_group: ProcessGroup,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        fp8_communication: bool = False,
        use_zbv: bool = False,
    ):
        assert tp_group is not None
        assert moe_dp_group is not None
        assert ep_group is not None

        # setup ep group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_group = ep_group
        self.fp8_communication = fp8_communication
        self.use_zbv = use_zbv

        if self.num_experts % self.ep_size != 0:
            raise ValueError("The number of experts must be divisible by the number of expert parallel groups.")

        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep
        held_experts = self.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]

        set_tensors_to_none(self.experts, exclude=set(held_experts))

        # setup moe_dp group
        self.moe_dp_group = moe_dp_group
        self.moe_dp_size = moe_dp_group.size()

        # setup global tp group
        self.tp_group = tp_group
        if self.tp_group.size() > 1:
            for expert in held_experts:
                expert.gate_proj = Linear1D_Col.from_native_module(
                    expert.gate_proj, self.tp_group, fp8_communication=self.fp8_communication, use_zbv=self.use_zbv
                )
                expert.up_proj = Linear1D_Col.from_native_module(
                    expert.up_proj, self.tp_group, fp8_communication=self.fp8_communication, use_zbv=self.use_zbv
                )
                expert.down_proj = Linear1D_Row.from_native_module(
                    expert.down_proj, self.tp_group, fp8_communication=self.fp8_communication, use_zbv=self.use_zbv
                )

        for p in self.experts.parameters():
            set_moe_tensor_ep_group(p, ep_group)

    @staticmethod
    def from_native_module(
        module: Qwen3MoeSparseMoeBlock,
        tp_group: ProcessGroup,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        *args,
        **kwargs,
    ) -> "EPQwen3MoeSparseMoeBlock":
        LazyInitContext.materialize(module)
        # layers in `mlp_only_layers` (or not on the `decoder_sparse_step`) keep a dense Qwen3MoeMLP
        if not isinstance(module, Qwen3MoeSparseMoeBlock):
            return module
        module.__class__ = EPQwen3MoeSparseMoeBlock
        fp8_communication = kwargs.get("fp8_communication", False)
        use_zbv = kwargs.get("use_zbv", False)
        module.setup_process_groups(tp_group, moe_dp_group, ep_group, fp8_communication, use_zbv)
        return module

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts = selected_experts.t().reshape(-1)
        selected_experts_idx = selected_experts.argsort()
        dispatch_states = hidden_states.repeat(self.top_k, 1)[selected_experts_idx]
        input_split_sizes = selected_experts.bincount(minlength=self.num_experts)

        output_split_sizes = torch.zeros_like(input_split_sizes)

        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)

        with torch.no_grad():
            activate_experts = output_split_sizes[: self.num_experts_per_ep].clone()
            for i in range(1, self.ep_size):
                activate_experts += output_split_sizes[i * self.num_experts_per_ep : (i + 1) * self.num_experts_per_ep]
            activate_experts = (activate_experts > 0).float()

        if self.fp8_communication:
            all_reduce_fp8(activate_experts, group=self.moe_dp_group)
        else:
            dist.all_reduce(activate_experts, group=self.moe_dp_group)

        input_split_list = input_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_split_list = output_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()

        output_states, _ = all_to_all_uneven(
            dispatch_states,
            input_split_list,
            output_split_list,
            self.ep_group,
            fp8_communication=self.fp8_communication,
        )
        # compute expert output
        output_states = EPGradScalerIn.apply(output_states, self.ep_size)
        if output_states.size(0) > 0:
            if self.num_experts_per_ep == 1:
                # no need to split
                expert = self.experts[self.expert_start_idx]
                output_states = DPGradScalerIn.apply(output_states, self.moe_dp_size, activate_experts[0])
                output_states = expert(output_states)
                output_states = DPGradScalerOut.apply(output_states, self.moe_dp_size, activate_experts[0])
            else:
                output_states_splits = output_states.split(output_split_sizes.tolist())
                output_states_list = []
                for i, split_states in enumerate(output_states_splits):
                    if split_states.size(0) == 0:
                        continue
                    expert = self.experts[self.expert_start_idx + i % self.num_experts_per_ep]
                    split_states = DPGradScalerIn.apply(
                        split_states, self.moe_dp_size, activate_experts[i % self.num_experts_per_ep]
                    )
                    split_states = expert(split_states)
                    split_states = DPGradScalerOut.apply(
                        split_states, self.moe_dp_size, activate_experts[i % self.num_experts_per_ep]
                    )
                    output_states_list.append(split_states)
                output_states = torch.cat(output_states_list)

        output_states = EPGradScalerOut.apply(output_states, self.ep_size)
        dispatch_states, _ = all_to_all_uneven(
            output_states, output_split_list, input_split_list, self.ep_group, fp8_communication=self.fp8_communication
        )

        recover_experts_idx = torch.empty_like(selected_experts_idx)
        recover_experts_idx[selected_experts_idx] = torch.arange(
            selected_experts_idx.size(0), device=selected_experts_idx.device
        )
        dispatch_states = dispatch_states[recover_experts_idx]
        k_hidden_states = dispatch_states.chunk(self.top_k)
        output_states = k_hidden_states[0] * routing_weights[:, 0, None]
        for i in range(1, self.top_k):
            output_states += k_hidden_states[i] * routing_weights[:, i, None]
        output_states = output_states.reshape(batch_size, sequence_length, hidden_dim)
        return output_states, router_logits


class Qwen3MoePipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Qwen3-MoE models
    under pipeline parallelism or sequence parallelism.
    """

    @staticmethod
    def qwen3_moe_model_forward(
        self: Qwen3MoeModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        past_router_logits: Optional[Tuple[torch.FloatTensor]] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        force_sp_output_gather: bool = True,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast, dict]:
        """Used for pipeline parallelism (``stage_manager`` is set) or sequence parallelism (``stage_manager`` is
        None, all the layers are run). The two are not supported together."""
        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_first_stage = stage_manager is None or stage_manager.is_first_stage()
        is_last_stage = stage_manager is None or stage_manager.is_last_stage()
        if stage_index is None:
            stage_index = [0, len(self.layers)]

        # retrieve input_ids and inputs_embeds
        if is_first_stage:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        else:
            batch_size, seq_length = hidden_states.shape[:-1]
        device = hidden_states.device

        # TODO: kv cache, attentions and hidden states are not recorded, same as the other pipeline forwards
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        sp_mode = shard_config.sequence_parallelism_mode if shard_config.enable_sequence_parallelism else None
        sp_size = shard_config.sequence_parallel_size
        sp_group = shard_config.sequence_parallel_process_group

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if cache_position is None:
            cache_position = torch.arange(seq_length, device=device)

        if shard_config.enable_flash_attention or sp_mode is not None:
            # the attention forward is replaced by `get_qwen3_flash_attention_forward` (see the policy),
            # which takes ColoAttention kwargs or a 4d additive causal mask
            if shard_config.enable_flash_attention:
                attention_mask = ColoAttention.prepare_attn_kwargs(
                    (batch_size, 1, seq_length, seq_length),
                    hidden_states.dtype,
                    hidden_states.device,
                    q_padding_mask=attention_mask,
                    is_causal=True,
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    hidden_states,
                    0,
                    sliding_window=self.config.sliding_window,
                )
        elif self.config._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.config._attn_implementation == "sdpa":
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), hidden_states, 0
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                0,
                sliding_window=self.config.sliding_window,
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if is_first_stage and sp_mode is not None:
            if is_share_sp_tp(sp_mode):
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, fp8_communication=shard_config.fp8_communication
                )
            elif sp_mode == "all_to_all":
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, 1 / sp_size, fp8_communication=shard_config.fp8_communication
                )

        all_router_logits = () if output_router_logits else None
        start_idx, end_idx = stage_index[0], stage_index[1]
        for decoder_layer in self.layers[start_idx:end_idx]:
            layer_args = (
                hidden_states,
                attention_mask,
                position_ids,
                None,  # past_key_value
                output_attentions,
                output_router_logits,
                use_cache,
                cache_position,
                position_embeddings,
            )
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, *layer_args)
            else:
                layer_outputs = decoder_layer(*layer_args)
            hidden_states = layer_outputs[0]

            # dense layers (Qwen3MoeMLP) have no router logits
            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        if output_router_logits and past_router_logits is not None:
            all_router_logits = past_router_logits + all_router_logits

        if not is_last_stage:
            out = {"hidden_states": hidden_states}
            if output_router_logits:
                out["past_router_logits"] = all_router_logits
            return out

        hidden_states = self.norm(hidden_states)
        if sp_mode is not None:
            if (not shard_config.parallel_output) or force_sp_output_gather or is_share_sp_tp(sp_mode):
                hidden_states = gather_sp_output(hidden_states, shard_config)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_router_logits] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            router_logits=all_router_logits,
        )

    @staticmethod
    def qwen3_moe_for_causal_lm_forward(
        self: Qwen3MoeForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        past_router_logits: Optional[Tuple[torch.FloatTensor]] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
        **kwargs,
    ):
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = Qwen3MoePipelineForwards.qwen3_moe_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            return_dict=True,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            past_router_logits=past_router_logits,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        if stage_manager is not None and not stage_manager.is_last_stage():
            return outputs

        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits, self.num_experts, self.num_experts_per_tok, attention_mask
            )
            if labels is not None:
                # make sure to reside in the same device
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,)
            if output_router_logits:
                output = (aux_loss,) + output + (outputs.router_logits,)
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            router_logits=outputs.router_logits,
        )
