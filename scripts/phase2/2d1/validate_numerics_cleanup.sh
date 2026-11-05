#!/usr/bin/env bash
# Container-only CPU regression and repeated reproduction of the failed CI case.
set -Eeuo pipefail
cd /workspace/ColossalAI
python="${COLOSSAL_PYTHON:-/home/wzj/venv/bin/python}"
mode="${1:-targeted}"
repeats="${REPEATS:-10}"
root="${RESULTS_ROOT:-/results/2d1}"
mkdir -p "$root"
output="$(mktemp -d "$root/numerics-cleanup-${mode}-XXXXXX")"
exec > >(tee "$output/run.log") 2>&1
trap 'status=$?; printf "exit_code=%s\n" "$status"; exit "$status"' EXIT
printf 'Results: %s\n' "$output"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 FAST_MODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/workspace/ColossalAI
git rev-parse HEAD > "$output/head.txt"
git -c core.fileMode=false diff --binary > "$output/change.patch"
sha256sum tests/kit/numerics/scenarios/{data_parallel,resume}.py \
    tests/test_numerics/test_numerics_data_parallel.py > "$output/source.sha256"
"$python" -c 'import sys, torch, transformers; print(sys.version); print("Torch:", torch.__version__); print("Transformers:", transformers.__version__); assert not torch.cuda.is_available()'
case "$mode" in
    targeted)
        timeout --signal=TERM --kill-after=15s 300s "$python" -m pytest -v -ra --tb=short \
            tests/test_numerics/test_numerics_data_parallel.py \
            tests/test_numerics/test_numerics_resume.py \
            --junitxml="$output/tests.xml"
        ;;
    repeat)
        for ((i=1; i<=repeats; i++)); do
            printf '\nDDP repeat %s/%s\n' "$i" "$repeats"
            timeout --signal=TERM --kill-after=15s 120s "$python" -m pytest -v -ra --tb=short \
                tests/test_numerics/test_numerics_data_parallel.py::test_ddp_fp32_equivalent \
                --junitxml="$output/repeat-$i.xml"
        done
        ;;
    cpu-gate)
        COLOSSAL_PYTHON="$python" COLOSSAL_RESULTS="$output" bash .github/scripts/ci/run_cpu_unit_tests.sh
        ;;
    *)
        printf 'Usage: %s {targeted|repeat|cpu-gate}\n' "$0" >&2
        exit 2
        ;;
esac
git diff --check
printf 'VALIDATION_PASSED\n'
