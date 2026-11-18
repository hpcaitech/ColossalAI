# Pinned Ouro test fixtures

These are verbatim source/config/model-card files from ByteDance's Apache-2.0
Ouro repositories. No pretrained tensor weights are included.

- Ouro-1.4B: https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1
- Ouro-2.6B: https://huggingface.co/ByteDance/Ouro-2.6B/tree/1ed04250da1a9936042725d302e81c8fa2ab5abd

Each manifest records the exact revision and SHA256 of every copied upstream
file. The offline loader checks them before execution. Original copyright
notices remain in the Python sources; the original model cards declare the
license. See LICENSE for the Apache 2.0 terms.

Do not format or adapt these files. Compatibility changes belong in
colossalai/shardformer/modeling/ouro.py. Regenerate fixtures from the fixed
offline snapshots with scripts/phase2/2d4/prepare_fixtures.py.
