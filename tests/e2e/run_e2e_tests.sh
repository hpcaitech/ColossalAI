#!/usr/bin/env bash
# ==============================================================================
# Canonical E2E Test Suite Runner for Cosmos3 Ascend 910B2 Training Optimization
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKSPACE_ROOT}"

# Prepend local .venv/bin if present
if [ -d "${WORKSPACE_ROOT}/.venv/bin" ]; then
    export PATH="${WORKSPACE_ROOT}/.venv/bin:${PATH}"
fi

TIER="all"
VERBOSE=0
FAIL_FAST=0
REPORT_JSON=""

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --tier {1|2|3|4|all}   Run specific test tier (default: all)"
    echo "  -v, --verbose          Enable verbose pytest output"
    echo "  -x, --fail-fast        Stop immediately on first test failure"
    echo "  --report-json <path>   Export test execution report to JSON file"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Tiers:"
    echo "  1: Tier 1 - Isolated Feature Coverage (105 tests across 21 features)"
    echo "  2: Tier 2 - Boundary, Limit & Negative Cases (105 tests across 21 features)"
    echo "  3: Tier 3 - Pairwise Cross-Feature Combinations (21 tests)"
    echo "  4: Tier 4 - Real-World End-to-End Application Scenarios (11 tests)"
    echo "  all: Complete 4-Tier Test Suite (242 tests)"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            TIER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -x|--fail-fast)
            FAIL_FAST=1
            shift
            ;;
        --report-json)
            REPORT_JSON="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"

PYTEST_ARGS=("-ra")
if [ "$VERBOSE" -eq 1 ]; then
    PYTEST_ARGS+=("-v")
else
    PYTEST_ARGS+=("-q")
fi

if [ "$FAIL_FAST" -eq 1 ]; then
    PYTEST_ARGS+=("-x")
fi

if [ -n "$REPORT_JSON" ]; then
    PYTEST_ARGS+=("--json-report" "--json-report-file=${REPORT_JSON}")
fi

echo "=============================================================================="
echo "Cosmos3 Ascend 910B2 Training Optimization E2E Test Suite Runner"
echo "Workspace:   ${WORKSPACE_ROOT}"
echo "Target Tier: ${TIER}"
echo "Pytest:      $(which pytest 2>/dev/null || echo 'Not in PATH')"
echo "Python:      $(which python3 2>/dev/null || echo 'Not in PATH')"
echo "=============================================================================="

TARGET_FILES=()
case "$TIER" in
    1)
        TARGET_FILES=("tests/e2e/test_tier1_feature_coverage.py")
        ;;
    2)
        TARGET_FILES=("tests/e2e/test_tier2_boundary_corner.py")
        ;;
    3)
        TARGET_FILES=("tests/e2e/test_tier3_combinations.py")
        ;;
    4)
        TARGET_FILES=("tests/e2e/test_tier4_real_world.py")
        ;;
    all)
        TARGET_FILES=(
            "tests/e2e/test_tier1_feature_coverage.py"
            "tests/e2e/test_tier2_boundary_corner.py"
            "tests/e2e/test_tier3_combinations.py"
            "tests/e2e/test_tier4_real_world.py"
        )
        ;;
    *)
        echo "Error: Invalid tier '${TIER}'. Must be one of: 1, 2, 3, 4, all" >&2
        exit 1
        ;;
esac

pytest "${PYTEST_ARGS[@]}" "${TARGET_FILES[@]}"
EXIT_CODE=$?

echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "E2E Test Suite Status: PASSED (Exit Code: 0)"
    echo "All selected tests executed successfully with 100% compliance."
else
    echo "E2E Test Suite Status: FAILED (Exit Code: ${EXIT_CODE})"
fi
echo "=============================================================================="
exit $EXIT_CODE
