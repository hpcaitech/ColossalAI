#!/usr/bin/env bash
set -euo pipefail
lslocks --output COMMAND,PID,TYPE,MODE,PATH | grep -E 'COMMAND|colossalai-ci-gpu'
ps -eo pid,ppid,etime,args | grep -E 'run_targeted|run_full_weights|run_host.sh|flock -w|test_ouro.py' | grep -v grep
