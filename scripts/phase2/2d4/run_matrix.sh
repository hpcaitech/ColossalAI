#!/usr/bin/env bash
# Host-side bounded matrix; a failing case is retained and does not hide later results.
set -uo pipefail
D2D4_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
D2D4_LOGROOT=/mnt/beegfs/ColossalAI/wangzhijian/containers/dev-torch213-tf517-peft0212/results/2d4
D2D4_MATRIX="$(mktemp -d "$D2D4_LOGROOT/matrix-$(date -u +%Y%m%d-%H%M%S)-XXXXXX")"
D2D4_FAILURE=0
D2D4_NUMBER=0
run_case() {
    D2D4_NUMBER=$((D2D4_NUMBER+1))
    local case_log="$D2D4_MATRIX/case-$D2D4_NUMBER.log"
    printf 'START %s %s\n' "$D2D4_NUMBER" "$*"
    if bash "$D2D4_DIR/run_host.sh" "$@" >"$case_log" 2>&1; then
        local status=0
    else
        local status=$?
        D2D4_FAILURE=1
    fi
    printf '%s\t%s\t%s\n' "$D2D4_NUMBER" "$status" "$*" >>"$D2D4_MATRIX/status.tsv"
    grep -E '^Results:|^RESULT|tensors aligned|AssertionError:|RuntimeError:' "$case_log" | tail -5
    printf 'END %s exit=%s\n' "$D2D4_NUMBER" "$status"
}
for loops in 2 4; do
    for plugin in zero1_fp32 zero2_fp32 zero2_bf16; do
        run_case --scenario independent --model ouro14 --loops "$loops" --plugin "$plugin" --world-size 2 --steps 3
    done
done
run_case --scenario independent --model ouro14 --loops 2 --plugin gemini_bf16 --world-size 2 --steps 3
for plugin in zero2_bf16 gemini_bf16; do
    run_case --scenario independent --model ouro26 --loops 4 --plugin "$plugin" --world-size 2 --steps 3
    for micro in 2 4; do
        run_case --scenario dp --model ouro14 --loops 4 --plugin "$plugin" --world-size 2 --micro-batches "$micro"
    done
    run_case --scenario dp_tokens --model ouro14 --loops 4 --plugin "$plugin" --world-size 2 --micro-batches 2
done
for plugin in zero2_fp32 zero2_bf16 gemini_bf16; do
    run_case --scenario resume --model ouro14 --loops 4 --plugin "$plugin" --world-size 2
done
printf 'MATRIX %s failed=%s\n' "$D2D4_MATRIX" "$D2D4_FAILURE"
exit "$D2D4_FAILURE"
