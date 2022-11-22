#!/usr/bin/env sh

for i in $(seq 0 7); do
    python convert_ckpt.py $1 $2 ${i} &
done

wait $(jobs -p)
