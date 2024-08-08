set -xe
pip install -r requirements.txt

BS=4
for PLUGIN in "torch_ddp" "torch_ddp_fp16" "low_level_zero" "gemini"
do
for GPUNUM in 1 4
do

echo "=============================================="
cat /opt/conda/envs/pytorch/lib/python3.9/site-packages/colossalai/booster/plugin/__init__.py
echo "=============================================="
cat /opt/conda/envs/pytorch/lib/python3.9/site-packages/colossalai/booster/plugin/moe_checkpoint.py
echo "=============================================="
cat /opt/conda/envs/pytorch/lib/python3.9/site-packages/colossalai/booster/plugin/moe_hybrid_parallel_plugin.py
echo "=============================================="
cat /opt/conda/envs/pytorch/lib/python3.9/site-packages/colossalai/booster/plugin/moe_hybrid_parallel_plugin.py
echo "=============================================="
cat /opt/conda/envs/pytorch/lib/python3.9/site-packages/colossalai/checkpoint_io/__init__.py
echo "=============================================="
cat /opt/conda/envs/pytorch/lib/python3.9/site-packages/colossalai/checkpoint_io/moe_checkpoint.py

colossalai run \
  --nproc_per_node ${GPUNUM} \
  --master_port 29505 \
  opt_benchmark.py \
  --model_name_or_path "facebook/opt-125m" \
  --plugin ${PLUGIN} \
  --batch_size ${BS}

done
done
