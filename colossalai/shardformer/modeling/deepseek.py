import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.functional as F
from torch.distributed import ProcessGroup
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.utils import is_flash_attn_2_available, logging

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
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_forward_split_backward,
    linear_with_async_comm,
    split_forward_gather_backward,
)
from colossalai.shardformer.layer.linear import Linear1D_Col, Linear1D_Row, ParallelModule
from colossalai.shardformer.shard import ShardConfig
from colossalai.shardformer.shard.utils import set_tensors_to_none
from colossalai.tensor.d_tensor.api import shard_rowwise, sharded_tensor_to_existing_param
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_group


# copied from modeling_deepseek.py
class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class EPDeepseekMoE(ParallelModule):
    def __init__(self):
        raise RuntimeError(f"Please use `from_native_module` to create an instance of {self.__class__.__name__}")

    def setup_process_groups(
        self,
        tp_group: ProcessGroup,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        fp8_communication: bool = False,
    ):
        assert tp_group is not None
        assert moe_dp_group is not None
        assert ep_group is not None

        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.num_experts = self.config.n_routed_experts
        assert self.num_experts % self.ep_size == 0
        self.fp8_communication = fp8_communication

        self.ep_group = ep_group
        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep
        held_experts = self.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]

        set_tensors_to_none(self.experts, exclude=set(held_experts))

        # setup moe_dp group
        self.moe_dp_group = moe_dp_group
        self.moe_dp_size = moe_dp_group.size()

        # setup tp group
        self.tp_group = tp_group
        if self.tp_group.size() > 1:
            for expert in held_experts:
                expert.gate_proj = Linear1D_Col.from_native_module(
                    expert.gate_proj, self.tp_group, fp8_communication=self.fp8_communication
                )
                expert.up_proj = Linear1D_Col.from_native_module(
                    expert.up_proj, self.tp_group, fp8_communication=self.fp8_communication
                )
                expert.down_proj = Linear1D_Row.from_native_module(
                    expert.down_proj, self.tp_group, fp8_communication=self.fp8_communication
                )

        for p in self.experts.parameters():
            set_moe_tensor_ep_group(p, ep_group)

        if self.config.n_shared_experts is not None:
            self.shared_experts.gate_proj = Linear1D_Col.from_native_module(
                self.shared_experts.gate_proj, self.tp_group, fp8_communication=self.fp8_communication
            )

            self.shared_experts.up_proj = Linear1D_Col.from_native_module(
                self.shared_experts.up_proj, self.tp_group, fp8_communication=self.fp8_communication
            )

            self.shared_experts.down_proj = Linear1D_Row.from_native_module(
                self.shared_experts.down_proj, self.tp_group, fp8_communication=self.fp8_communication
            )

    @staticmethod
    def from_native_module(
        module,
        tp_group: ProcessGroup,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        *args,
        **kwargs,
    ) -> "EPDeepseekMoE":
        LazyInitContext.materialize(module)
        if module.__class__.__name__ == "DeepseekMLP":
            return module
        module.__class__ = EPDeepseekMoE
        fp8_communication = kwargs.get("fp8_communication", False)
        module.setup_process_groups(tp_group, moe_dp_group, ep_group, fp8_communication=fp8_communication)
        return module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_experts_idx, topk_experts_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # [t0, t1, t2 ...]
        hidden_states = hidden_states.repeat_interleave(
            self.num_experts_per_tok, dim=0
        )  # after repeat_interleave: [t0 t0 t1 t1 t2 t2 ... ]

        flat_topk_experts_idx = topk_experts_idx.view(-1)  # [e0 e1 e2 ...]
        # The elements of flat_topk_token_idx are token ids, which are arranged in ascending order of expert ids.
        flat_topk_token_idx = flat_topk_experts_idx.argsort()

        # Now we adjust the order of the hidden states, also in ascending order of expert id
        dispatch_states = hidden_states[flat_topk_token_idx]
        input_split_sizes = flat_topk_experts_idx.bincount(minlength=self.num_experts)  # [n0, n1, n2, n3]
        output_split_sizes = torch.zeros_like(input_split_sizes)

        # [n0, n1, n2, n3] [m0, m1, m2, m3] -> [n0, n1, m0, m1] [n2, n3, m2, m3]
        dist.all_to_all_single(
            output_split_sizes,
            input_split_sizes,
            group=self.ep_group,
        )

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
        output_states = EPGradScalerIn.apply(output_states, self.ep_size)

        if output_states.size(0) > 0:
            if self.num_experts_per_ep == 1:
                expert = self.experts[self.expert_start_idx]
                output_states = DPGradScalerIn.apply(output_states, self.moe_dp_size, activate_experts[0])
                output_states = expert(output_states)
                output_states = DPGradScalerOut.apply(output_states, self.moe_dp_size, activate_experts[0])
            else:
                output_states_splits = output_states.split(output_split_sizes.tolist())
                output_states_list = []
                for i, split_states in enumerate(output_states_splits):
                    if split_states.size(0) == 0:  # no token routed to this experts
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
        recover_token_idx = torch.empty_like(flat_topk_token_idx)
        recover_token_idx[flat_topk_token_idx] = torch.arange(
            flat_topk_token_idx.size(0), device=flat_topk_token_idx.device
        )

        output_hidden_states = dispatch_states[recover_token_idx]  # t0 t0 t1 t1 t2 t2
        output_hidden_states = output_hidden_states.view(-1, self.num_experts_per_tok, orig_shape[-1])
        output_hidden_states = (output_hidden_states * topk_experts_weight[:, :, None]).sum(dim=-2)  # (B*S, h)
        output_hidden_states = output_hidden_states.view(*orig_shape)
        output_hidden_states = AddAuxiliaryLoss.apply(output_hidden_states, aux_loss)
        if self.config.n_shared_experts is not None:
            output_hidden_states = output_hidden_states + self.shared_experts(identity)
        return output_hidden_states


class DeepseekMoEGate_Col(ParallelModule):
    def parallel_linear(self, hidden_states):
        assert (
            hidden_states.shape[-1] == self.weight.shape[-1]
        ), "Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.".format(
            hidden_states.shape, self.weight.shape, self.weight.shape[-1]
        )

        output = linear_with_async_comm(
            hidden_states, self.weight, None, self.process_group, True, fp8_communication=self.fp8_communication
        )

        # All-gather across the partitions.
        output = gather_forward_split_backward(
            output, dim=-1, process_group=self.process_group, fp8_communication=self.fp8_communication
        )
        return output

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = self.parallel_linear(hidden_states)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss

    @staticmethod
    def from_native_module(
        module, process_group: ProcessGroup, config, gather_output, fp8_communication
    ) -> "DeepseekMoEGate_Col":
        LazyInitContext.materialize(module)
        module.process_group = process_group
        module.fp8_communication = fp8_communication
        sharded_weight = shard_rowwise(module.weight.data, process_group)
        sharded_tensor_to_existing_param(sharded_weight, module.weight)
        module.__class__ = DeepseekMoEGate_Col
        return module


class DeepseekPipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Llama models
    under pipeline setting.
    """

    @staticmethod
    def deepseek_model_forward(
        self: "DeepseekModel",
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if stage_manager.is_first_stage():
            # retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions, for the first stage, hidden_states is the input embeddings,
        # for the other stages, hidden_states is the output of the previous stage
        if is_flash_attn_2_available():
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        start_idx, end_idx = stage_index[0], stage_index[1]
        for idx, decoder_layer in enumerate(self.layers[start_idx:end_idx], start=start_idx):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if stage_manager.is_last_stage():
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None

        if stage_manager.is_last_stage():
            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        # always return dict for imediate stage
        return {
            "hidden_states": hidden_states,
        }

    @staticmethod
    def deepseek_for_causal_lm_forward(
        self: "DeepseekForCausalLM",
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MixtralForCausalLM

        >>> model = DeepseekForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        logger = logging.get_logger(__name__)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = DeepseekPipelineForwards.deepseek_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
        )
        past_key_values = None

        if stage_manager.is_last_stage():
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=outputs[0],
                attentions=None,
            )
        else:
            out = {}
            hidden_states = outputs.get("hidden_states")
            out["hidden_states"] = hidden_states
            return out


def get_deepseek_flash_attention_forward(shard_config, sp_mode=None, sp_size=None, sp_group=None):
    logger = logging.get_logger(__name__)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if sp_mode is not None:
            assert sp_mode in ["all_to_all", "split_gather", "ring"], "Invalid sp_mode"
            assert (sp_size is not None) and (
                sp_group is not None
            ), "Must specify sp_size and sp_group for sequence parallel"

        # DeepseekFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # sp: modify sp_len when sequence parallel mode is ring
        if sp_mode in ["split_gather", "ring"]:
            q_len *= sp_size

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # sp: all-to-all comminucation when introducing sequence parallel
        if sp_mode == "all_to_all":
            query_states = all_to_all_comm(query_states, sp_group, fp8_communication=shard_config.fp8_communication)
            key_states = all_to_all_comm(key_states, sp_group, fp8_communication=shard_config.fp8_communication)
            value_states = all_to_all_comm(value_states, sp_group, fp8_communication=shard_config.fp8_communication)
            bsz, q_len, _ = query_states.size()
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=0
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )
        # sp: all-to-all comminucation when introducing sequence parallel
        if sp_mode == "all_to_all":
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()  # (1, 8, 128)
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2, fp8_communication=shard_config.fp8_communication
            )  # (1, 4, 256)
        else:
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def get_deepseek_flash_attention_model_forward(shard_config, sp_mode=None, sp_size=None, sp_group=None):
    logger = logging.get_logger(__name__)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # TODO: upgrade transformers to 4.44.0 to fix the bug, remove the hard code.
        self._use_flash_attention_2 = shard_config.enable_flash_attention
        self._use_sdpa = False if shard_config.enable_flash_attention else self._use_sdpa

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        if sp_mode in ["ring", "split_gather"]:
            inputs_embeds = split_forward_gather_backward(
                inputs_embeds, 1, sp_group, fp8_communication=shard_config.fp8_communication
            )
        elif sp_mode == "all_to_all":
            inputs_embeds = split_forward_gather_backward(
                inputs_embeds, 1, sp_group, 1 / sp_size, fp8_communication=shard_config.fp8_communication
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if sp_mode == "ring" or sp_mode == "split_gather":
            hidden_states = gather_forward_split_backward(
                hidden_states, 1, sp_group, fp8_communication=shard_config.fp8_communication
            )
        elif sp_mode == "all_to_all":
            hidden_states = gather_forward_split_backward(
                hidden_states, 1, sp_group, grad_scale=sp_size, fp8_communication=shard_config.fp8_communication
            )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return forward
