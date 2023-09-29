# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .tokenizer import build_tokenizer

_TOKENIZER = None
_PADDED_VOCAB_SIZE = -1


def initialize_tokenizer(vocab_file, tokenizer_type, vocab_extra_ids=0):
    tokenizer, padded_vocab_size = build_tokenizer(vocab_file, tokenizer_type, vocab_extra_ids)
    global _TOKENIZER, _PADDED_VOCAB_SIZE
    _TOKENIZER = tokenizer
    _PADDED_VOCAB_SIZE = padded_vocab_size


def get_tokenizer():
    global _TOKENIZER
    return _TOKENIZER


def get_padded_vocab_size():
    global _PADDED_VOCAB_SIZE
    return _PADDED_VOCAB_SIZE
