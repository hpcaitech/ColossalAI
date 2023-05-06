#!/bin/bash
set -xe

# FIXME(ver217): only run bert finetune to save time

cd glue_bert && bash ./test_ci.sh && cd ..
