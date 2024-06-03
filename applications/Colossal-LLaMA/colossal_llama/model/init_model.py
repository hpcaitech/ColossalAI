#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialize new model with updated tokenizer by calculating the mean values from original model
"""
import argparse

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_model_and_tokenizer_path",
        type=str,
        required=True,
        default=None,
        help="Source path of model & tokenizer",
    )
    parser.add_argument("--target_tokenizer_path", type=str, required=True, default=None, help="Target tokenizer path")
    parser.add_argument("--target_model_path", type=str, required=True, default=None, help="Target model path")
    args = parser.parse_args()

    source_tokenizer = LlamaTokenizer.from_pretrained(args.source_model_and_tokenizer_path)
    source_tokenizer.add_bos_token = False
    source_tokenizer.add_eos_token = False
    if source_tokenizer.pad_token is None:
        source_tokenizer.pad_token = source_tokenizer.unk_token
    source_vocab = source_tokenizer.get_vocab()

    target_tokenizer = LlamaTokenizer.from_pretrained(args.target_tokenizer_path)
    target_tokenizer.add_bos_token = False
    target_tokenizer.add_eos_token = False
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.unk_token
    target_vocab = target_tokenizer.get_vocab()
    target_inverted_vocab = {v: k for k, v in target_vocab.items()}

    assert len(target_vocab) > len(
        source_vocab
    ), f"Target vocab size({len(target_vocab)}) must be greater than source vocab size({len(source_vocab)})"

    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    source_model = LlamaForCausalLM.from_pretrained(args.source_model_and_tokenizer_path)
    source_model.eval()
    source_model = source_model.to(gpu_device)

    source_input_embeddings = source_model.get_input_embeddings()
    assert isinstance(source_input_embeddings, torch.nn.Embedding)
    assert source_input_embeddings.weight.shape[0] == len(source_vocab)
    source_input_embeddings.eval()

    source_output_embeddings = source_model.get_output_embeddings()
    assert isinstance(source_output_embeddings, torch.nn.Linear)
    assert source_output_embeddings.bias is None
    assert source_output_embeddings.weight.shape[0] == len(source_vocab)
    source_output_embeddings.eval()

    input_embeddings = source_input_embeddings.weight.cpu().detach().numpy()
    output_embeddings = source_output_embeddings.weight.cpu().detach().numpy()
    for i in range(len(source_vocab), len(target_vocab)):
        if i % 500 == 0:
            logger.info(f"processing {i}/{len(target_vocab)} target tokens")
        target_token = target_inverted_vocab[i]
        target_to_source_token_ids = torch.LongTensor(source_tokenizer([target_token])["input_ids"][0])
        target_to_source_token_ids = target_to_source_token_ids.to(gpu_device)

        target_to_source_input_embedding = (
            source_input_embeddings.weight[target_to_source_token_ids]
            .mean(dim=0)
            .unsqueeze(dim=0)
            .cpu()
            .detach()
            .numpy()
        )
        target_to_source_output_embedding = (
            source_output_embeddings.weight[target_to_source_token_ids]
            .mean(dim=0)
            .unsqueeze(dim=0)
            .cpu()
            .detach()
            .numpy()
        )

        input_embeddings = np.concatenate((input_embeddings, target_to_source_input_embedding), axis=0)
        output_embeddings = np.concatenate((output_embeddings, target_to_source_output_embedding), axis=0)

    source_model = source_model.to(cpu_device)
    assert isinstance(source_model, LlamaForCausalLM)

    # expand
    source_model.resize_token_embeddings(new_num_tokens=len(target_vocab))
    source_model.model.embed_tokens.weight.data = torch.Tensor(input_embeddings)
    source_model.lm_head.weight.data = torch.Tensor(output_embeddings)

    source_model = source_model.half()
    source_model.save_pretrained(save_directory=args.target_model_path)


if __name__ == "__main__":
    main()
