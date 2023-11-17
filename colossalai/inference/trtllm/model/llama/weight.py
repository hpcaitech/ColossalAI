# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import configparser
import time
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from safetensors import safe_open

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for LLaMA model

    Returns a dictionary of scaling factors for the selected layers of the
    LLaMA model.

    Args:
        model_path (str): Path to the quantized LLaMA model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        LLaMA model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_output' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
            'gate_act': gate_act_scale,
            'gate_weights': gate_weights_scale,
            'proj_act': proj_act_scale,
            'proj_weights': proj_weights_scale,
        }
    """

    if model_path is None:
        logger.warning(f"--quantized_fp8_model_path not specified. "
                       f"Initialize quantization scales automatically.")
        return get_dummy_quant_scales(num_layers)
    weight_dict = np.load(model_path)

    # yapf: disable
    scaling_factor = {
        'qkv_act': [],
        'qkv_weights': [],
        'qkv_output': [],
        'dense_act': [],
        'dense_weights': [],
        'fc_act': [],
        'fc_weights': [],
        'gate_act': [],
        'gate_weights': [],
        'proj_act': [],
        'proj_weights': [],
    }

    for layer in range(num_layers):
        scaling_factor['qkv_act'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:activation_scaling_factor'].item()
            ))
        scaling_factor['qkv_weights'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:weights_scaling_factor'].item()
            ))
        if quant_mode is not None and quant_mode.has_fp8_kv_cache():
            # Not calibrarting KV cache.
            scaling_factor['qkv_output'].append(1.0)
        scaling_factor['dense_act'].append(weight_dict[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
        scaling_factor['dense_weights'].append(weight_dict[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
        scaling_factor['fc_act'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:activation_scaling_factor'].item())
        scaling_factor['fc_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:weights_scaling_factor'].item())
        scaling_factor['gate_act'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:activation_scaling_factor'].item())
        scaling_factor['gate_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:weights_scaling_factor'].item())
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
        f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size,
                  -1)[:, None, :, :].expand(num_head, reps, head_size,
                                            v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone()


def parse_ft_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('llama', 'hidden_size')
    n_head = gpt_config.getint('llama', 'num_attention_heads')
    n_layer = gpt_config.getint('llama', 'num_hidden_layers')
    n_positions = gpt_config.getint('llama', 'max_position_embeddings')
    vocab_size = gpt_config.getint('llama', 'vocab_size')
    hidden_act = gpt_config.get('llama', 'hidden_act')
    inter_size = gpt_config.getint('llama', 'intermediate_size', fallback=None)
    n_kv_head = gpt_config.getint('llama', 'num_key_value_heads', fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    return n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head


def load_from_hf_llama(tensorrt_llm_llama: tensorrt_llm.models.LLaMAForCausalLM,
                       hf_llama,
                       mapping=Mapping(),
                       dtype='float32'):
    tensorrt_llm.logger.info('Loading weights from HF LLaMA...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_llama.num_kv_heads
    mha_mode = (num_kv_heads == tensorrt_llm_llama.num_heads)

    model_params = dict(hf_llama.named_parameters())
    for l in range(hf_llama.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if not mha_mode:
            head_size = tensorrt_llm_llama.hidden_size // tensorrt_llm_llama.num_heads
            if num_kv_heads < mapping.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_kv_heads,
                                         mapping.tp_size)
                v_weight = dup_kv_weight(v_weight, num_kv_heads,
                                         mapping.tp_size)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight

    torch_dtype = str_dtype_to_torch(dtype)
    layers_per_pipeline_stage = hf_llama.config.num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            if tensorrt_llm_llama.use_parallel_embedding:
                v = split(v, mapping.tp_size, mapping.tp_rank,
                          tensorrt_llm_llama.embedding_sharding_dim)
            if mapping.is_first_pp_rank():
                tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
                    split(v, mapping.tp_size, mapping.tp_rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue
            idx = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
            if idx >= tensorrt_llm_llama.num_layers:
                continue
            if 'input_layernorm.weight' in k:
                tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].post_layernorm.weight
                dst.value = v
            elif 'self_attn.qkv_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].attention.qkv.weight
                if not mha_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], mapping.tp_size, mapping.tp_rank)
                    wk = split(v[1], mapping.tp_size, mapping.tp_rank)
                    wv = split(v[2], mapping.tp_size, mapping.tp_rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size),
                                              model_emb)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'self_attn.o_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].attention.dense.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.up_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.gate.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.down_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.gate_proj.weight' in k:
                dst = tensorrt_llm_llama.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type)
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.view(
                        dtype=torch.float32).numpy()
                    scales = tensorrt_llm_llama.layers[
                        idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_meta_llama(
        tensorrt_llm_llama: tensorrt_llm.models.LLaMAForCausalLM,
        meta_ckpt_dir,
        mapping=Mapping(),
        dtype="float32"):
    torch_dtype = str_dtype_to_torch(dtype)

    def gather_ckpts(ckpts):
        gathered = {}
        for k in ckpts[0]:
            d = 0
            if any([n in k for n in ["wo", "w2", "tok"]]):
                d = 1
            if "norm" in k or "rope" in k:  # no TP
                gathered[k] = ckpts[0][k].clone()
            else:
                gathered[k] = torch.cat([pt[k] for pt in ckpts], dim=d).clone()
        return gathered

    def split_ckpt(ckpt, ranks_per_ckpt, ckpt_rank):
        split_ckpt = {}
        for k in ckpt:
            d = 0
            if any([n in k for n in ["wo", "w2", "tok"]]):
                d = 1
            if "norm" in k or "rope" in k:  # no TP
                split_ckpt[k] = ckpt[k].clone()
            elif tensorrt_llm_llama.num_kv_heads < mapping.tp_size and any(
                [n in k for n in ["wk", "wv"]]):
                assert mapping.tp_size % tensorrt_llm_llama.num_kv_heads == 0
                # special case: we need to duplicate KV head
                tmp = dup_kv_weight(ckpt[k], tensorrt_llm_llama.num_kv_heads,
                                    mapping.tp_size)
                split_ckpt[k] = torch.split(tmp,
                                            tmp.shape[d] // ranks_per_ckpt,
                                            dim=d)[ckpt_rank].clone()
            else:
                split_ckpt[k] = torch.split(ckpt[k],
                                            ckpt[k].shape[d] // ranks_per_ckpt,
                                            dim=d)[ckpt_rank].clone()
        return split_ckpt

    def get_current_weights(num_ckpts):
        if num_ckpts > mapping.tp_size:
            # combine ckpts
            assert (num_ckpts % mapping.tp_size) == 0
            nf = num_ckpts // mapping.tp_size
            fs = nf * mapping.tp_rank
            file_ids = list(range(fs, fs + nf))
            ckpts = []
            for f in file_ids:
                ckpt = torch.load(Path(meta_ckpt_dir,
                                       f"consolidated.{f:02d}.pth"),
                                  map_location="cpu")
                ckpts.append(ckpt)
            return gather_ckpts(ckpts)
        elif num_ckpts < mapping.tp_size:
            # split ckpt
            assert (mapping.tp_size % num_ckpts) == 0
            ranks_per_ckpt = mapping.tp_size // num_ckpts
            ckpt_fid = mapping.tp_rank // ranks_per_ckpt
            ckpt_rank = mapping.tp_rank % ranks_per_ckpt
            nH_per_ckpt = tensorrt_llm_llama.num_heads // num_ckpts
            assert (nH_per_ckpt % ranks_per_ckpt) == 0
            ckpt = torch.load(Path(meta_ckpt_dir,
                                   f"consolidated.{ckpt_fid:02d}.pth"),
                              map_location="cpu")
            return split_ckpt(ckpt, ranks_per_ckpt, ckpt_rank)

        # num_ckpts == tensor_parallel, 1:1 mapping from files to TP
        return torch.load(Path(meta_ckpt_dir,
                               f"consolidated.{mapping.tp_rank:02d}.pth"),
                          map_location="cpu")

    def permute(w, nH, d, dH):
        # due to MQA's wk, nH*dH != d could be true
        return w.view(nH, dH // 2, 2, d).transpose(1, 2).reshape(nH * dH, d)

    if not hasattr(load_from_meta_llama, "saved_embed"):
        load_from_meta_llama.saved_embed = None

    def gather_embedding(cur_embed, name: str, num_ckpts):
        if mapping.tp_size == 1:
            # even if num_ckpts > 1, get_current_weights will already have it gathered
            return cur_embed
        if load_from_meta_llama.saved_embed is None:
            embeds = [None] * num_ckpts
            for i in range(num_ckpts):
                ckpt = torch.load(Path(meta_ckpt_dir,
                                       f"consolidated.{i:02d}.pth"),
                                  map_location="cpu")
                embeds[i] = ckpt[name]
            embed = torch.cat(embeds, dim=1).to(torch_dtype)
            load_from_meta_llama.saved_embed = torch_to_numpy(
                embed)  # cache the embedding, not needed if no refit
        return load_from_meta_llama.saved_embed

    tensorrt_llm.logger.info('Loading weights from Meta LLaMA checkpoints ...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        torch.int8
    elif quant_mode.is_int4_weight_only():
        torch.quint4x2
    quant_mode.is_weight_only()
    num_kv_heads = tensorrt_llm_llama.num_kv_heads
    mha_mode = (num_kv_heads == tensorrt_llm_llama.num_heads)

    ckpts = list(Path(meta_ckpt_dir).glob("consolidated.*.pth"))
    num_ckpts = len(ckpts)
    # llama/llama2 doesn't have MQA. So, simplifying loader logic by not worrying about it.
    assert num_kv_heads > 1 or num_kv_heads >= num_ckpts, \
        f"We don't know how the {num_kv_heads} KV heads are distributed among {num_ckpts} checkpoints."

    head_size = tensorrt_llm_llama.hidden_size // tensorrt_llm_llama.num_heads
    ckpt = get_current_weights(num_ckpts)
    layers_range = list(
        range(mapping.pp_rank * tensorrt_llm_llama.num_layers,
              (mapping.pp_rank + 1) * tensorrt_llm_llama.num_layers, 1))

    for l in layers_range:
        prefix = f'layers.{l}.attention.'
        q_weight = permute(ckpt[prefix + 'wq.weight'].clone(),
                           nH=(tensorrt_llm_llama.num_heads // mapping.tp_size),
                           d=tensorrt_llm_llama.hidden_size,
                           dH=head_size)
        if num_kv_heads < mapping.tp_size and num_ckpts >= mapping.tp_size:
            assert mapping.tp_size % num_kv_heads == 0
            assert False, "Not supported yet"
        k_weight = permute(ckpt[prefix + 'wk.weight'].clone(),
                           nH=((num_kv_heads + mapping.tp_size - 1) //
                               mapping.tp_size),
                           d=tensorrt_llm_llama.hidden_size,
                           dH=head_size)
        v_weight = ckpt[prefix + 'wv.weight'].clone()

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        ckpt[prefix + 'qkv.weight'] = qkv_weight

    for k, v in ckpt.items():
        v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if "tok_embeddings" in k:
            if not tensorrt_llm_llama.use_parallel_embedding:
                v = gather_embedding(v, k, num_ckpts)
            elif tensorrt_llm_llama.embedding_sharding_dim == 0:
                # this needs a gather and then resplit along different dims
                v = gather_embedding(v, k, num_ckpts)
                v = split(v, mapping.tp_size, mapping.tp_rank, 0)
            if mapping.is_first_pp_rank():
                tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif "output" in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.lm_head.weight.value = v
        elif k == "norm.weight":
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.ln_f.weight.value = v
        else:
            # layer specific weights
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(
                layer_idx) - mapping.pp_rank * tensorrt_llm_llama.num_layers
            if idx >= tensorrt_llm_llama.num_layers:
                continue
            if 'attention_norm.weight' in k:
                tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
            elif 'ffn_norm.weight' in k:
                tensorrt_llm_llama.layers[idx].post_layernorm.weight.value = v
            elif 'feed_forward.w3.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.gate.weight.value = v
            elif 'feed_forward.w2.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.proj.weight.value = v
            elif 'feed_forward.w1.weight' in k:
                tensorrt_llm_llama.layers[idx].mlp.fc.weight.value = v
            elif 'attention.wo.weight' in k:
                tensorrt_llm_llama.layers[idx].attention.dense.weight.value = v
            elif 'attention.qkv.weight' in k:
                tensorrt_llm_llama.layers[idx].attention.qkv.weight.value = v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_binary(tensorrt_llm_llama: LLaMAForCausalLM,
                     dir_path,
                     mapping=Mapping(),
                     fp16=False,
                     multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))

    n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head = parse_ft_config(
        Path(dir_path) / 'config.ini')
    np_dtype = np.float16 if fp16 else np.float32

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def set_smoothquant_scale_factors(module,
                                      pre_scale_weight,
                                      dir_path,
                                      basename,
                                      shape,
                                      per_tok_dyn,
                                      per_channel,
                                      is_qkv=False,
                                      rank=None):
        suffix = "bin"
        if per_channel:
            if rank is not None:
                suffix = f"{rank}." + suffix
            suffix = "col." + suffix

        col_shape = shape if (per_channel or is_qkv) else [1, 1]

        if per_tok_dyn:
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            if is_qkv and not per_channel:
                t = fromfile(dir_path,
                             f"{basename}scale_w_quant_orig.{rank}.{suffix}",
                             col_shape, np.float32)
            else:
                t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                             col_shape, np.float32)
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                         np.float32)
            pre_scale_weight.value = t
            if is_qkv:
                t = fromfile(dir_path,
                             f"{basename}scale_y_accum_quant.{rank}.{suffix}",
                             col_shape, np.float32)
            else:
                t = fromfile(dir_path,
                             f"{basename}scale_y_accum_quant.{suffix}",
                             col_shape, np.float32)
            module.per_channel_scale.value = t
            t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                         np.float32)
            module.act_scale.value = t

    def set_smoother(module, dir_path, base_name, shape, rank):
        suffix = f"{rank}.bin"
        t = fromfile(dir_path, f"{base_name}.smoother.{suffix}", shape,
                     np.float32)
        module.smoother.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_llama, "quant_mode", QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    def sq_trick(x):
        return x.view(np.float32) if use_smooth_quant else x

    # Debug
    suffix = gen_suffix(mapping.tp_rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    if mapping.is_first_pp_rank():
        tensorrt_llm_llama.vocab_embedding.weight.value = (fromfile(
            dir_path, 'vocab_embedding.weight.bin', [vocab_size, n_embd]))

    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.ln_f.weight.value = (fromfile(
            dir_path, 'ln_f.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'lm_head.weight.bin',
                              [vocab_size, n_embd])

    if vocab_size % mapping.tp_size != 0:
        # padding
        vocab_size_padded = tensorrt_llm_llama.lm_head.out_features * mapping.tp_size
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, mapping.tp_size, mapping.tp_rank))

    layers_range = list(
        range(mapping.pp_rank * tensorrt_llm_llama.num_layers,
              (mapping.pp_rank + 1) * tensorrt_llm_llama.num_layers, 1))

    for i in layers_range:
        n_groups = n_head // n_kv_head
        c_attn_out_dim = (
            3 * n_embd // mapping.tp_size) if not multi_query_mode else (
                n_embd // mapping.tp_size +
                (n_embd // n_head * n_groups) // mapping.tp_size * 2)
        idx = i - mapping.pp_rank * tensorrt_llm_llama.num_layers
        tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
        t = fromfile(
            dir_path, 'model.layers.' + str(i) +
            '.attention.query_key_value.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        if t is not None:
            dst = tensorrt_llm_llama.layers[idx].attention.qkv.weight
            if use_smooth_quant:
                dst.value = sq_trick(
                    np.ascontiguousarray(np.transpose(t, [1, 0])))
                set_smoothquant_scale_factors(
                    tensorrt_llm_llama.layers[idx].attention.qkv,
                    tensorrt_llm_llama.layers[idx].input_layernorm.scale_to_int,
                    dir_path,
                    'model.layers.' + str(i) + '.attention.query_key_value.',
                    [1, c_attn_out_dim],
                    quant_per_token_dyn,
                    quant_per_channel,
                    rank=mapping.tp_rank,
                    is_qkv=True)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    torch.tensor(t), plugin_weight_only_quant_type)
                # workaround for trt not supporting int8 inputs in plugins currently
                dst.value = processed_torch_weights.view(
                    dtype=torch.float32).numpy()
                scales = tensorrt_llm_llama.layers[
                    i].attention.qkv.per_channel_scale
                scales.value = torch_weight_scales.numpy()
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_llama.layers[idx].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
            [n_embd // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            dst.value = sq_trick(np.ascontiguousarray(np.transpose(t, [1, 0])))
            dense_scale = getattr(tensorrt_llm_llama.layers[idx].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attention.dense.',
                [1, n_embd], quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_llama.layers[idx].attention.dense,
                         dir_path,
                         'model.layers.' + str(i) + '.attention.dense',
                         [1, n_embd // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_llama.layers[
                i].attention.dense.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_llama.layers[idx].post_layernorm.weight
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.post_layernorm.weight.bin')

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.fc.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)

        if use_smooth_quant:
            tensorrt_llm_llama.layers[idx].mlp.fc.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].mlp.fc,
                tensorrt_llm_llama.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.fc.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_llama.layers[i].mlp.fc.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_llama.layers[i].mlp.fc.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_llama.layers[
                idx].mlp.fc.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.gate.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size], w_type)
        if use_smooth_quant:
            tensorrt_llm_llama.layers[idx].mlp.gate.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].mlp.gate,
                tensorrt_llm_llama.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.gate.',
                [1, inter_size // mapping.tp_size],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_llama.layers[i].mlp.gate.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_llama.layers[i].mlp.gate.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_llama.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.proj.weight.' + suffix,
                     [inter_size // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_llama.layers[idx].mlp.proj.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            proj_scale = getattr(tensorrt_llm_llama.layers[idx].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_llama.layers[idx].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.proj.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_llama.layers[idx].mlp.proj, dir_path,
                         'model.layers.' + str(i) + '.mlp.proj',
                         [1, inter_size // mapping.tp_size], mapping.tp_rank)
        elif use_weight_only:
            dst = tensorrt_llm_llama.layers[i].mlp.proj.weight
            processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                torch.tensor(t), plugin_weight_only_quant_type)
            # workaround for trt not supporting int8 inputs in plugins currently
            dst.value = processed_torch_weights.view(
                dtype=torch.float32).numpy()
            scales = tensorrt_llm_llama.layers[i].mlp.proj.per_channel_scale
            scales.value = torch_weight_scales.numpy()
        else:
            tensorrt_llm_llama.layers[idx].mlp.proj.weight.value = (
                np.ascontiguousarray(np.transpose(t, [1, 0])))

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_llama.layers[
                idx].attention.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_llama.layers[
                idx].attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')


def load_from_gptq_llama(tensorrt_llm_llama,
                         quant_ckpt_path,
                         mapping=Mapping(),
                         dtype="float16"):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise GPTQ LLaMA safetensors...')
    tik = time.time()

    if quant_ckpt_path.endswith(".safetensors"):
        groupwise_qweight_safetensors = safe_open(quant_ckpt_path,
                                                  framework="pt",
                                                  device=0)
        model_params = {
            key: groupwise_qweight_safetensors.get_tensor(key)
            for key in groupwise_qweight_safetensors.keys()
        }
    elif quant_ckpt_path.endswith(".pt"):
        model_params = torch.load(quant_ckpt_path,
                                  map_location=torch.device('cpu'))
    else:
        assert False, "Quantized checkpoint format not supported!"

    def unpack_int32_into_int8(w_packed):
        # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def preprocess_groupwise_weight_params(weight_name,
                                           qweight_int32=None,
                                           qzeros_int32=None,
                                           scales_fp16=None):
        if weight_name is not None:
            qweight_int32 = model_params[weight_name].cpu()
            qzeros_int32 = model_params[weight_name[:-7] + 'qzeros'].cpu()
            scales_fp16 = model_params[weight_name[:-7] + 'scales'].cpu()

        UINT4_TO_INT4_FLAG = 1
        GPTQ_FLAG = 1
        packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
        preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

        qweight_unpacked_int8 = unpack_int32_into_int8(
            qweight_int32.T).T.contiguous() - 8
        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2).view(torch.float32)
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)
        zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8 * UINT4_TO_INT4_FLAG -
                               GPTQ_FLAG) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        return qweight_interleaved.contiguous(), scales_fp16.contiguous(
        ), zeros_x_scales_fp16.contiguous()

    layer_ids = [
        extract_layer_idx(key) for key in groupwise_qweight_safetensors.keys()
    ]
    layer_ids = [
        int(layer_idx) for layer_idx in layer_ids if layer_idx is not None
    ]
    num_hidden_layers = max(layer_ids) + 1
    num_kv_heads = tensorrt_llm_llama.num_kv_heads
    mha_mode = (num_kv_heads == tensorrt_llm_llama.num_heads)
    suffixs = ['qweight', 'qzeros', 'scales']

    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        prefix = f'model.layers.{l}.self_attn.'
        split_qkv_suf = []

        for suf in suffixs:
            q_part = model_params[prefix + 'q_proj.' + suf].cpu()
            k_part = model_params[prefix + 'k_proj.' + suf].cpu()
            v_part = model_params[prefix + 'v_proj.' + suf].cpu()
            qkv_part = torch.cat([q_part, k_part, v_part], dim=0)
            dim = qkv_part.shape
            qkv_part = qkv_part.reshape(3, dim[0] // 3, dim[1])
            split_qkv = qkv_part.split(dim[1] // mapping.tp_size,
                                       dim=2)[mapping.tp_rank]
            split_qkv = torch.cat([
                split_qkv[0, :, :].squeeze(0), split_qkv[1, :, :].squeeze(0),
                split_qkv[2, :, :].squeeze(0)
            ],
                                  dim=1)
            split_qkv_suf.append(split_qkv)

        th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
            None, split_qkv_suf[0], split_qkv_suf[1], split_qkv_suf[2])

        idx = l - mapping.pp_rank * layers_per_pipeline_stage
        tensorrt_llm_llama.layers[
            idx].attention.qkv.qweight.value = th_qweight.numpy()
        tensorrt_llm_llama.layers[
            idx].attention.qkv.scale.value = th_zero.numpy()
        tensorrt_llm_llama.layers[
            idx].attention.qkv.zero.value = th_scale.numpy()

    torch_dtype = str_dtype_to_torch(dtype)

    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            if mapping.is_first_pp_rank():
                tensorrt_llm_llama.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_llama.lm_head.weight.value = np.ascontiguousarray(
                    split(v, mapping.tp_size, mapping.tp_rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx not in layers_range:
                continue
            idx = idx - mapping.pp_rank * layers_per_pipeline_stage

            if 'input_layernorm.weight' in k:
                tensorrt_llm_llama.layers[idx].input_layernorm.weight.value = v
            elif 'post_attention_layernorm.weight' in k:
                tensorrt_llm_llama.layers[idx].post_layernorm.weight.value = v
            elif 'self_attn.o_proj.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size,
                                      dim=0)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_llama.layers[
                    idx].attention.dense.qweight.value = th_qweight.numpy()
                tensorrt_llm_llama.layers[
                    idx].attention.dense.scale.value = th_zero.numpy()
                tensorrt_llm_llama.layers[
                    idx].attention.dense.zero.value = th_scale.numpy()
            elif 'mlp.up_proj.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size,
                                      dim=1)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_llama.layers[
                    idx].mlp.gate.qweight.value = th_qweight.numpy()
                tensorrt_llm_llama.layers[
                    idx].mlp.gate.scale.value = th_zero.numpy()
                tensorrt_llm_llama.layers[
                    idx].mlp.gate.zero.value = th_scale.numpy()
            elif 'mlp.down_proj.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size,
                                      dim=0)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_llama.layers[
                    idx].mlp.proj.qweight.value = th_qweight.numpy()
                tensorrt_llm_llama.layers[
                    idx].mlp.proj.scale.value = th_zero.numpy()
                tensorrt_llm_llama.layers[
                    idx].mlp.proj.zero.value = th_scale.numpy()
            elif 'mlp.gate_proj.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size,
                                      dim=1)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_llama.layers[
                    idx].mlp.fc.qweight.value = th_qweight.numpy()
                tensorrt_llm_llama.layers[
                    idx].mlp.fc.scale.value = th_zero.numpy()
                tensorrt_llm_llama.layers[
                    idx].mlp.fc.zero.value = th_scale.numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def load_from_awq_llama(tensorrt_llm_llama: LLaMAForCausalLM,
                        quant_ckpt_path,
                        mapping=Mapping(),
                        dtype="float16"):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise AWQ LLaMA safetensors...')
    tik = time.time()

    if quant_ckpt_path.endswith(".safetensors"):
        groupwise_qweight_safetensors = safe_open(quant_ckpt_path,
                                                  framework="pt",
                                                  device=0)
        awq_llama = {
            key: groupwise_qweight_safetensors.get_tensor(key)
            for key in groupwise_qweight_safetensors.keys()
        }
    elif quant_ckpt_path.endswith(".pt"):
        awq_llama = torch.load(quant_ckpt_path,
                               map_location=torch.device('cpu'))
    else:
        assert False, "Quantized checkpoint format not supported!"

    group_size = awq_llama["model.layers.0.self_attn.o_proj.weight"].numel(
    ) // awq_llama[
        "model.layers.0.self_attn.o_proj.weight_quantizer._amax"].numel()

    awq_llama_block_names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]

    tensorrt_llm_llama_block_names = [
        "input_layernorm.weight",
        "post_layernorm.weight",
    ]

    getattr(tensorrt_llm_llama, 'quant_mode', QuantMode(0))

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def AWQ_quantize_pack_preprocess(weight, scale):
        scale = scale.repeat_interleave(group_size, dim=0)
        weight = weight / scale
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = packer(qweight_int8.cpu())
        int4_weight = preprocessor(int4_weight, torch.quint4x2)
        return int4_weight.view(torch.float32).cpu().numpy()

    def process_and_assign_weight(awq_llama, mPrefix, mOp, tp_dim=0):
        weight = awq_llama[mPrefix + ".weight"].T.contiguous()
        [k, n] = weight.shape
        weight = weight.split(weight.shape[tp_dim] // mapping.tp_size,
                              dim=tp_dim)[mapping.tp_rank]
        amax = awq_llama[mPrefix + ".weight_quantizer._amax"].reshape(
            (n, int(k / group_size))).T.contiguous()
        amax = amax.split(amax.shape[tp_dim] // mapping.tp_size,
                          dim=tp_dim)[mapping.tp_rank]
        pre_quant_scale = awq_llama[
            mPrefix + ".input_quantizer._pre_quant_scale"].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = pre_quant_scale.split(k // mapping.tp_size,
                                                    dim=1)[mapping.tp_rank]
        scale = amax / 8.0
        mOp.qweight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.scale.value = scale.to(torch_dtype).cpu().numpy()
        mOp.pre_quant_scale.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    def deSmooth(weight, pre_quant_scale):
        [k, n] = weight.shape
        pre_quant_scale = pre_quant_scale.repeat(
            (n, 1)).transpose(1, 0).contiguous()
        weight = weight * pre_quant_scale
        return weight

    def reSmooth(weight, pre_quant_scale):
        [k, n] = weight.shape
        pre_quant_scale = pre_quant_scale.repeat(
            (n, 1)).transpose(1, 0).contiguous()
        weight = weight / pre_quant_scale
        return weight

    def get_scale(weight):
        weight = weight.T.contiguous()
        [n, k] = weight.shape
        weight = weight.reshape(n, int(k / group_size), group_size)
        weight = torch.abs(weight.reshape(-1, group_size))
        amax, idx = weight.max(1)
        amax = amax.reshape(n, int(k / group_size)).T.contiguous()
        return amax / 8

    def reSmooth_and_get_scale(weight, pre_quant_scale, avg_pre_quant_scale):
        weight = deSmooth(weight, pre_quant_scale)
        weight = reSmooth(weight, avg_pre_quant_scale)
        scale = get_scale(weight)
        return weight, scale

    def process_and_assign_qkv_weight(awq_llama, prefix, mOp):
        q_weight = awq_llama[prefix + "self_attn.q_proj.weight"].T.contiguous()
        k_weight = awq_llama[prefix + "self_attn.k_proj.weight"].T.contiguous()
        v_weight = awq_llama[prefix + "self_attn.v_proj.weight"].T.contiguous()
        k = q_weight.shape[0]

        q_weight = q_weight.split(q_weight.shape[1] // mapping.tp_size,
                                  dim=1)[mapping.tp_rank]
        k_weight = k_weight.split(k_weight.shape[1] // mapping.tp_size,
                                  dim=1)[mapping.tp_rank]
        v_weight = v_weight.split(v_weight.shape[1] // mapping.tp_size,
                                  dim=1)[mapping.tp_rank]

        q_pre_quant_scale = awq_llama[
            prefix +
            "self_attn.q_proj.input_quantizer._pre_quant_scale"].reshape((1, k))
        k_pre_quant_scale = awq_llama[
            prefix +
            "self_attn.k_proj.input_quantizer._pre_quant_scale"].reshape((1, k))
        v_pre_quant_scale = awq_llama[
            prefix +
            "self_attn.v_proj.input_quantizer._pre_quant_scale"].reshape((1, k))

        qkv_pre_quant_scale = (q_pre_quant_scale + k_pre_quant_scale +
                               v_pre_quant_scale) / 3.0
        q_weight, q_scale = reSmooth_and_get_scale(q_weight, q_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        k_weight, k_scale = reSmooth_and_get_scale(k_weight, k_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        v_weight, v_scale = reSmooth_and_get_scale(v_weight, v_pre_quant_scale,
                                                   qkv_pre_quant_scale)

        qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
        qkv_scale = torch.cat((q_scale, k_scale, v_scale), dim=1)

        mOp.pre_quant_scale.value = qkv_pre_quant_scale.to(
            torch_dtype).cpu().numpy()
        mOp.qweight.value = AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale)
        mOp.scale.value = qkv_scale.to(torch_dtype).cpu().numpy()

    # Check if we need to pad vocab
    v = awq_llama.get('model.embed_tokens.weight')
    [vocab_size, k] = v.shape
    pad_vocab = False
    pad_vocab_size = vocab_size
    if vocab_size % 64 != 0:
        pad_vocab = True
        pad_vocab_size = int((vocab_size + 63) / 64) * 64
    if pad_vocab:
        new_v = torch.zeros([pad_vocab_size, k])
        new_v[:vocab_size, :] = v
        v = new_v
    if mapping.is_first_pp_rank():
        tensorrt_llm_llama.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    layer_ids = [extract_layer_idx(key) for key in awq_llama.keys()]
    layer_ids = [
        int(layer_idx) for layer_idx in layer_ids if layer_idx is not None
    ]

    num_hidden_layers = max(layer_ids) + 1
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for layer_idx in layers_range:
        prefix = "model.layers." + str(layer_idx) + "."
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        for idx, awq_attr in enumerate(awq_llama_block_names):
            v = awq_llama[prefix + awq_attr]
            layer = attrgetter(tensorrt_llm_llama_block_names[idx])(
                tensorrt_llm_llama.layers[layer_idx])
            setattr(layer, 'value', v.to(torch_dtype).cpu().numpy())

        # Attention QKV Linear
        # concatenate the Q, K, V layers weights.
        process_and_assign_qkv_weight(
            awq_llama, prefix,
            tensorrt_llm_llama.layers[layer_idx].attention.qkv)

        # Attention Dense (out_proj) Linear
        mPrefix = prefix + "self_attn.o_proj"
        mOp = tensorrt_llm_llama.layers[layer_idx].attention.dense
        process_and_assign_weight(awq_llama, mPrefix, mOp, 0)

        # MLP up_proj (mlp.gate) Linear
        mPrefix = prefix + "mlp.up_proj"
        mOp = tensorrt_llm_llama.layers[layer_idx].mlp.gate
        process_and_assign_weight(awq_llama, mPrefix, mOp, 1)

        # MLP down_proj (mlp.proj) Linear
        mPrefix = prefix + "mlp.down_proj"
        mOp = tensorrt_llm_llama.layers[layer_idx].mlp.proj
        process_and_assign_weight(awq_llama, mPrefix, mOp, 0)

        # MLP gate_proj (mlp.fc) Linear
        mPrefix = prefix + "mlp.gate_proj"
        mOp = tensorrt_llm_llama.layers[layer_idx].mlp.fc
        process_and_assign_weight(awq_llama, mPrefix, mOp, 1)

    v = awq_llama['model.norm.weight']
    if mapping.is_last_pp_rank():
        tensorrt_llm_llama.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    #lm_head
    if pad_vocab:
        weight = awq_llama['lm_head.weight']
        [vocab_size, k] = weight.shape
        new_weight = torch.zeros([pad_vocab_size, k])
        new_weight[:vocab_size, :] = weight
        new_weight = new_weight.T.contiguous()
        amax = awq_llama['lm_head.weight_quantizer._amax'].reshape(
            [vocab_size, k // group_size])
        new_amax = torch.ones([pad_vocab_size, k // group_size])
        new_amax[:vocab_size, :] = amax
        new_amax = new_amax.T.contiguous()
        new_scale = new_amax / 8
        tensorrt_llm_llama.lm_head.qweight.value = AWQ_quantize_pack_preprocess(
            new_weight, new_scale)
        tensorrt_llm_llama.lm_head.scale.value = new_scale.to(
            torch_dtype).cpu().numpy()
        tensorrt_llm_llama.lm_head.pre_quant_scale.value = awq_llama[
            'lm_head.input_quantizer._pre_quant_scale'].to(
                torch_dtype).cpu().numpy()
    else:
        mPrefix = "lm_head"
        mOp = tensorrt_llm_llama.lm_head
        if mapping.is_last_pp_rank():
            process_and_assign_weight(awq_llama, mPrefix, mOp, 1)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
