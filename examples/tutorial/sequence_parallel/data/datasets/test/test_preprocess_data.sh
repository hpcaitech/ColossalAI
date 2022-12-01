#!/bin/bash

IMPL=cached
python ../preprocess_data.py \
       --input test_samples.json \
       --vocab vocab.txt \
       --dataset-impl ${IMPL} \
       --output-prefix test_samples_${IMPL} \
       --workers 1 \
       --log-interval 2
