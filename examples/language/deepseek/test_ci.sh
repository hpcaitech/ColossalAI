#!/usr/bin/env bash
set -euxo pipefail

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun --standalone --nproc_per_node=2 smoke_test.py
