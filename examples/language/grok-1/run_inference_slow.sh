#!/usr/bin/env bash

PRETRAINED=${1:-"hpcaitech/grok-1"}
TOKENIZER=${2:-"tokenizer.model"}

python3 inference.py --pretrained "$PRETRAINED" \
    --tokenizer "$TOKENIZER" \
    --max_new_tokens 64 \
    --text "The company's annual conference, featuring keynote speakers and exclusive product launches, will be held at the Los Angeles Convention Center from October 20th to October 23rd, 2021. Extract the date mentioned in the above sentence." \
            "将以下句子翻译成英语。 我喜欢看电影和读书。" \
            "All books have the same weight, 10 books weigh 5kg, what is the weight of 2 books?"
