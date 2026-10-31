#!/usr/bin/env bash
set -euo pipefail

CI_SKIP_ISSUE="https://github.com/hpcaitech/ColossalAI/issues/6453"
message="SKIP: the RoBERTa example requires external data, tokenizer/config assets, multi-host setup, and legacy ColossalAI APIs. Tracking issue: ${CI_SKIP_ISSUE}"

echo "${message}"
if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
    echo "::warning title=RoBERTa example CI skipped::${message}"
fi

exit 0
