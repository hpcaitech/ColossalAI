#!/usr/bin/env bash
# Run inside the development container; never change its existing development venv.
set -Eeuo pipefail
root="${RESULTS_ROOT:-/results/2d1}"
mkdir -p "$root"
output="$(mktemp -d "$root/numerics-torch251-setup-XXXXXX")"
exec > >(tee "$output/setup.log") 2>&1
trap 'status=$?; printf "exit_code=%s\n" "$status"; exit "$status"' EXIT
venv="${NUMERICS_CPU_VENV:-/home/wzj/venvs/numerics-torch251-cpu}"
printf 'Results: %s\nVenv: %s\n' "$output" "$venv"
if [[ ! -x "$venv/bin/python" ]]; then
    /home/wzj/venv/bin/python -m venv "$venv"
fi
"$venv/bin/python" -m pip install --timeout 60 --retries 2 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$venv/bin/python" -m pip install --timeout 60 --retries 2 --index-url "${NUMERICS_PYPI_INDEX:-https://pypi.org/simple}" -r /workspace/ColossalAI/requirements/requirements.txt pytest "torch==2.5.1+cpu"
"$venv/bin/python" -m pip install --no-deps -e /workspace/ColossalAI
"$venv/bin/python" -c 'import sys, torch, transformers; print(sys.version); print(torch.__version__, transformers.__version__); assert torch.__version__ == "2.5.1+cpu"; assert torch.version.cuda is None'
"$venv/bin/python" -m pip freeze > "$output/requirements.txt"
