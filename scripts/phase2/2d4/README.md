# 2D4 Ouro validation

The formal tests exercise the current mounted checkout and its current
tests/kit/numerics implementation. See [RESULTS.md](RESULTS.md) for dated
results and [RESULTS-20261104.md](RESULTS-20261104.md) for the original failures.

## Environment and logs

- Server checkout: /mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI
- Container checkout: /workspace/ColossalAI
- Validation container: wzj-colossalai-2d4-20261118, physical H20 GPUs 5/6.
- Container GPU indices: 0/1; container uses the existing code/home/cache/results mounts.
- Torch 2.13.0+cu130, Transformers 5.17.0.
- Persistent logs, JSON reports, JUnit and tensor traces: /results/2d4/.
- Host results: /mnt/beegfs/ColossalAI/wangzhijian/containers/dev-torch213-tf517-peft0212/results/2d4/.

No separate development clone or historical numerical-tool checkout is used.
Each runner records HEAD, source hashes and the actual failure status through tee.

## Reproduce

On the server, run_host.sh holds the shared CI GPU lock, checks occupancy,
and enters the container. Override D2D4_GPU_LOCK if the CI lock path changes.

~~~bash
export D2D4_CONTAINER=wzj-colossalai-2d4-20261118 D2D4_GPUS=5,6
bash /mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI/scripts/phase2/2d4/run_host.sh preflight
D2D4_OFFLINE_FIXTURES=1 bash /mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI/scripts/phase2/2d4/run_host.sh tests tests/test_booster/test_plugin/test_ouro.py tests/test_checkpoint_io/test_gemini_offload_export.py -q
~~~

Inside the container, after reserving its GPUs:

~~~bash
cd /workspace/ColossalAI
# CPU reference/checkpoint/lazy/HF reload checks, with an empty HF cache.
D2D4_OFFLINE_FIXTURES=1 bash scripts/phase2/2d4/run_tests.sh tests/test_model_zoo/test_ouro.py -q

# Full tiny-model GPU acceptance and export regression.
D2D4_OFFLINE_FIXTURES=1 bash scripts/phase2/2d4/run_tests.sh tests/test_booster/test_plugin/test_ouro.py tests/test_checkpoint_io/test_gemini_offload_export.py -q

# Shared Gemini checkpoint regression.
bash scripts/phase2/2d4/run_tests.sh tests/test_checkpoint_io/test_gemini_master_resume.py -q

# Original-size official weights; uses the existing shared cache.
D2D4_TIMEOUT=900 bash scripts/phase2/2d4/run.sh full-weights --model ouro14
D2D4_TIMEOUT=900 bash scripts/phase2/2d4/run.sh full-weights --model ouro26
~~~

D2D4_OFFLINE_FIXTURES=1 creates fresh empty hub/module caches in the run
directory. It proves the source-only pinned fixtures suffice; it does not
erase the user's cache. Full-weight smoke tests require the original snapshots.

## What is checked

- Both pinned Ouro-1.4B/Ouro-2.6B code snapshots; tiny = 2 decoder layers,
  hidden size 64, vocabulary 128, batch 2, sequence 16.
- Every layer is called 2/4 times per forward, using the same Parameters.
  Hooks also require actual recomputation when checkpointing is enabled.
- Candidate goes through the ColossalAI adapter and real ShardFormer native
  policy, then the selected Booster plugin.
- Independent tiny reference constructs the official model with a temporary
  HF5 RoPE constructor bridge and builds its own triangular padding mask.
  Official decoder, loop, gate and loss are unchanged on both sides.
- The shared tool compares FP64 truth, matching-precision PyTorch reference,
  and candidate: logits, actual LM loss, all 27 parameter gradients, post-step
  weights and a three-step synthetic projected-loss trajectory. Gradients of
  both exit-gate parameters must be finite and nonzero.
- Gemini/FSDP gradient capture uses the shared tool's Adam beta1=0 first-moment
  method on both sides. Judge thresholds are unchanged.
- DDP/ZeRO/FSDP: FP32 and BF16. Gemini: default gathered and explicit sharded
  BF16. CPU offload: ZeRO-2 and static fully sharded Gemini BF16.
- Real causal CE with unequal valid-token counts: two GPU ranks accumulating
  four global micro-batches must match one GPU accumulating the same batches.
- Resume: 2+2 steps vs four uninterrupted steps, bitwise training-state/loss/
  parameter equality, plus default adapter-class HF reload.
  The fixture is deterministic; caller-managed dataloader position/RNG for a
  stochastic application is not supplied by Booster checkpoints.
- Lazy Init + checkpoint + real CE training on one/two GPUs.
- Unmodified official configs that omit pad_token_id load with no padding index;
  token zero stays trainable. Distributed tests retry only address-in-use errors.
- CPU-offloaded Gemini exports immediately after step, with no refreshing
  forward: state_dict plus sharded/unsharded files equal current master values,
  and exporting leaves the working chunk's placement/sync state unchanged.

## Boundaries

This is eager full attention with use_cache=False, no sliding window, default
full-head RoPE and contiguous positions. Checkpointing is non-reentrant for
every layer, without activation offload. ShardFormer TP/PP/SP/EP, fused kernels,
FP8, generation and other offload recipes are not covered.

Original-size smoke tests load all official weights strictly, compare forward
logits/loss bitwise with the official loop and independent mask, then execute
one single-GPU BF16 ZeRO-2 backward/update with checkpoint recomputation.
They are not full-size FP64 gradient L2, long-context, convergence or performance
acceptance. Tiny tests do not establish these properties.

## Historical probes

run.sh --scenario ... remains useful for reproducing the old plugin probes.
Those use the same adapted model as plain-PyTorch reference and therefore
isolate plugin behavior. They are not the independent official reference used
by the new formal tests. Historical aliases differ: gemini_bf16 means explicit
sharding; gemini_bf16_default means default chunk search.
