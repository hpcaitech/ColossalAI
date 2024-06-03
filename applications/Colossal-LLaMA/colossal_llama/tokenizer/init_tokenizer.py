#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Initialize new tokenizer for continual pre-training
"""

import argparse
import json
import os
from typing import List, Union

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from colossalai.logging import get_dist_logger

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

logger = get_dist_logger()


def expand_vocab_tokenizer(
    source_tokenizer_dir: Union[str, os.PathLike], target_tokenizer_dir: Union[str, os.PathLike], new_tokens: List[str]
) -> None:
    """Expand tokenizer for continue pre-training."""
    if os.path.exists(target_tokenizer_dir):
        raise RuntimeError(f"Find existed directory {target_tokenizer_dir}")

    source_tokenizer = LlamaTokenizer.from_pretrained(source_tokenizer_dir)
    logger.info(source_tokenizer)
    source_sp_processor = source_tokenizer.sp_model
    source_spm = sp_pb2_model.ModelProto()
    source_spm.ParseFromString(source_sp_processor.serialized_model_proto())

    logger.info(f"Source tokenizer size: {len(source_sp_processor)}")

    # Add new tokens to source tokenizer.
    source_spm_tokens = set([p.piece for p in source_spm.pieces])
    for piece in new_tokens:
        assert isinstance(piece, str), f"Invalid token({piece}) type {type(piece)}"
        if piece in source_spm_tokens:
            # Skip existed token.
            continue
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        source_spm.pieces.append(new_p)
    logger.info(f"Expand vocab from {len(source_spm_tokens)} to {len(source_spm.pieces)}")

    # Save
    os.makedirs(target_tokenizer_dir)
    target_tokenizer_model_path = os.path.join(target_tokenizer_dir, "tokenizer.model")
    with open(file=target_tokenizer_model_path, mode="wb") as fp:
        fp.write(source_spm.SerializeToString())

    target_tokenizer = LlamaTokenizer(vocab_file=target_tokenizer_model_path)
    target_tokenizer.save_pretrained(save_directory=target_tokenizer_dir)
    logger.info(f"Successfully save expand tokenizer to {target_tokenizer_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_tokenizer_dir", type=str, required=True, default=None, help="Source tokenizer directory"
    )
    parser.add_argument(
        "--target_tokenizer_dir", type=str, required=True, default=None, help="Target tokenizer directory"
    )
    parser.add_argument(
        "--expand_tokens_file",
        type=str,
        required=True,
        default=None,
        help="Path of the file containing tokens to be extended",
    )
    args = parser.parse_args()

    expand_tokens = []
    with open(file=args.expand_tokens_file, mode="r", encoding="utf-8") as fp_reader:
        for line in fp_reader:
            item = json.loads(line)
            # e.g., {"piece": "你好"}
            token = item["piece"]
            if token in expand_tokens:
                continue
            expand_tokens.append(token)
    expand_tokens.sort(key=lambda t: len(t), reverse=False)

    expand_vocab_tokenizer(
        source_tokenizer_dir=args.source_tokenizer_dir,
        target_tokenizer_dir=args.target_tokenizer_dir,
        new_tokens=expand_tokens,
    )


if __name__ == "__main__":
    main()
