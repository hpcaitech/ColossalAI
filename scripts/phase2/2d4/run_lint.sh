#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ColossalAI
export PRE_COMMIT_HOME=/home/wzj/cache/pre-commit PYTHONDONTWRITEBYTECODE=1
out="$(mktemp -d /results/2d4/lint-20261118-XXXXXX)"
printf 'Results: %s\n' "$out"
python -m pre_commit run --files .pre-commit-config.yaml .github/scripts/ci/run_colossalai_batch.sh colossalai/shardformer/modeling/ouro.py colossalai/shardformer/policies/_remote.py colossalai/shardformer/policies/ouro.py colossalai/zero/gemini/utils.py tests/kit/remote_models.py tests/kit/ouro.py tests/kit/model_zoo/transformers/__init__.py tests/kit/model_zoo/transformers/ouro.py tests/kit/numerics/capture/recipes.py tests/test_model_zoo/test_ouro.py tests/test_booster/test_plugin/test_ouro.py tests/test_checkpoint_io/test_gemini_offload_export.py tests/fixtures/ouro/NOTICE.md tests/fixtures/ouro/LICENSE scripts/phase2/2d4/*.py scripts/phase2/2d4/*.sh --show-diff-on-failure 2>&1 | tee "$out/run.log"
