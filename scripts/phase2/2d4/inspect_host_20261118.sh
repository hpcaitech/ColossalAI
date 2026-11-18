#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI
results=/mnt/beegfs/ColossalAI/wangzhijian/containers/dev-torch213-tf517-peft0212/results/2d4
mkdir -p "$results"
run="$(mktemp -d "$results/inspect-20261118-XXXXXX")"
exec > >(tee "$run/run.log") 2>&1
printf 'Results: %s\n' "$run"
date -Is
id
cd "$repo"
pwd
git remote -v
git status --short --branch
git rev-parse HEAD
find . -name .git -prune -o -name AGENTS.md -print
docker ps -a --filter name=wzj-colossalai --format '{{.Names}} {{.Status}} {{.Image}}'
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
for name in wzj-colossalai-2d4 wzj-colossalai-dev-tf517; do
  docker inspect --format '{{.Name}} user={{.Config.User}} running={{.State.Running}} image={{.Image}} mounts={{json .Mounts}} devices={{json .HostConfig.Devices}} device_requests={{json .HostConfig.DeviceRequests}}' "$name"
done
