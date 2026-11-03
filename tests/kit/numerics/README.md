# Numerical alignment

A shared tool for checking that an implementation (a fused kernel, a sharded module, a plugin) computes the same thing
as a reference implementation, allowing only the error that the precision itself explains.

## The rule: baseline error

Run the same inputs and weights three times:

| Run | What runs | Role |
| --- | --- | --- |
| `truth` | reference implementation in float64 | ground truth |
| `ref` | reference implementation at the tested precision | measures the **baseline error**: how far correct code drifts from the truth at this precision |
| `test` | implementation under test, same precision recipe as `ref` | what is being checked |

For every tensor (outputs, input gradients, parameter gradients):

```
baseline error = RMSE(ref,  truth)
test error     = RMSE(test, truth)
pass if  test error <= multiplier * baseline error + tol / 10
```

`multiplier` is 3 for fp16 / bf16 and 2 for fp32, and at least 3 for tensors with fewer than 1000 elements. A `test`
that is already `allclose` to `ref` (`atol = rtol = tol`, default `1e-4`) passes directly. This is the rule PyTorch
uses to check `torch.compile` across its model suites (`same()` in `torch/_dynamo/utils.py`).

Why not a fixed tolerance: the error a correct bf16 implementation makes differs by orders of magnitude between models
and tensors. A tolerance tight enough for one model rejects another; a loose one hides real bugs. The baseline error
is measured per tensor, so each tensor gets its own allowance.

## Quick start: an operator

Check that `scaled_dot_product_attention` matches a plain attention written from the formula:

```python
import torch
import torch.nn.functional as F

from tests.kit.numerics import assert_aligned, run_fwd_bwd


def eager_attention(q, k, v):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    return torch.softmax(scores, dim=-1) @ v


def sdpa_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


g = torch.Generator().manual_seed(0)
inputs = {name: torch.randn(2, 4, 128, 64, generator=g, dtype=torch.float64) for name in ("q", "k", "v")}

truth = run_fwd_bwd(eager_attention, inputs, torch.float64)
ref = run_fwd_bwd(eager_attention, inputs, torch.bfloat16)
test = run_fwd_bwd(sdpa_attention, inputs, torch.bfloat16)
assert_aligned(test, ref, truth)
```

`run_fwd_bwd(fn, inputs, dtype)` casts the floating-point inputs to `dtype`, runs `fn(**inputs)`, backpropagates a
fixed random upstream gradient (the same in all three runs) and returns a `Trace` with `out/...` for the outputs and
`grad/<input name>` for the input gradients. TF32 is disabled during the run.

## A module

Pass an `nn.Module` instead of a function. It is deep-copied and cast for each run, so the caller's module is not
modified and every run starts from the same weights. Parameter gradients are recorded as `grad/param/<name>`, so the
module under test must use the same parameter names; load the reference weights into it:

```python
class SplitKLinear(torch.nn.Linear):
    """Same math as nn.Linear, summed in two halves like a sharded kernel would."""

    def forward(self, x):
        h = self.in_features // 2
        return x[..., :h] @ self.weight[:, :h].t() + x[..., h:] @ self.weight[:, h:].t() + self.bias


torch.manual_seed(0)
ref_module = torch.nn.Linear(256, 128)
test_module = SplitKLinear(256, 128)
test_module.load_state_dict(ref_module.state_dict())

inputs = {"input": torch.randn(64, 256, dtype=torch.float64)}
truth = run_fwd_bwd(ref_module, inputs, torch.float64)
ref = run_fwd_bwd(ref_module, inputs, torch.bfloat16)
test = run_fwd_bwd(test_module, {"x": inputs["input"]}, torch.bfloat16)
```

Here the two modules name their input differently (`input` vs `x`), so the input gradients would not match by name.
Rename or drop entries before judging; `Trace(tensors, meta)` builds a new trace from a dict:

```python
from tests.kit.numerics import Trace, judge

test = Trace({name.replace("grad/x", "grad/input"): t for name, t in test.items()}, test.meta)
print(judge(test, ref, truth))
```

## Reading the report

`judge(test, ref, truth)` returns an `AlignmentReport`; `assert_aligned` raises `AssertionError` with the same table.
The two reports below come from the linear-layer cases in `tests/test_numerics`. A correct reordering of the
summation:

```
3/3 tensors aligned
tensor                                                    baseline  test_err    ratio  verdict
out                                                       4.74e-02  5.58e-02     1.18  PASS
grad/w                                                    2.34e-02  2.34e-02     1.00  PASS
grad/x                                                    3.27e-02  3.27e-02     1.00  PASS
```

A sharding bug that drops one column of the reduction dimension:

```
0/3 tensors aligned
tensor                                                    baseline  test_err    ratio  verdict
grad/w                                                    2.34e-02  5.02e-01    21.49  FAIL (> 3x baseline error)
out                                                       4.74e-02  9.81e-01    20.70  FAIL (> 3x baseline error)
grad/x                                                    3.27e-02  6.72e-01    20.52  FAIL (> 3x baseline error)
```

`baseline` and `test_err` are RMSE against the float64 truth, `ratio` is their quotient. Failures are listed first,
largest ratio first. Other verdicts: `missing in test` / `missing in truth` (names differ), `shape ...`,
`non-finite test output`, and `non-float tensors differ` for integer tensors, which must match exactly. Each entry
also carries `metrics` with max abs error, relative L2 error and cosine similarity. A comparison of no tensors at all
fails (`<all>: nothing was compared`).

The judges only compare the tensors the runs recorded, so a parameter that gets no gradient on every side would go
unnoticed. Each run therefore lists the parameters that require a gradient but got none in
`meta["missing_grads"]`, and the scenarios fail them (`no gradient in ...`, via `require_gradients`). A parameter
that legitimately gets none, such as a branch the inputs never reach, must be declared: `unused_params` of
`check_fwd_bwd`, or `NumericsSpec(unused_params=...)` for a model_zoo entry. Frozen parameters are not expected to
have a gradient.

## Comparing runs from different machines

A `Trace` is a read-only mapping from tensor name to tensor plus a `meta` dict (dtype, function, torch and
transformers versions, anything passed as keyword arguments). It saves to a single `.safetensors` file with the
metadata as JSON in the header, so a run on one machine can be judged against runs from another:

```python
truth.save("truth.safetensors")  # e.g. on the GPU machine
ref.save("ref.safetensors")
test.save("test.safetensors")  # e.g. on the NPU machine

report = judge(Trace.load("test.safetensors"), Trace.load("ref.safetensors"), Trace.load("truth.safetensors"))
```

`Trace.capture(outputs, **meta)` turns any tensor, nested dict / list / tuple or HF `ModelOutput` into a trace.

## Things that make the comparison meaningless

- **Different precision recipes.** `ref` and `test` must reduce precision in the same places. Autocast (fp32 weights,
  bf16 matmuls, fp32 softmax and norms) has a smaller baseline error than casting the whole model to bf16, so judging
  a whole-cast `test` against an autocast `ref` reports failures in correct code.
- **Sharing code between `ref` and `test`.** A bug in shared code appears in both and is never caught. The reference
  must be an independent implementation (e.g. HF eager modeling code), not a call into the code under test.
- **A reference that cannot run in float64.** Fused kernels such as flash attention usually have no fp64 path; use a
  plain implementation as the reference and test the fused one.
- **Randomness inside the function.** Generate noise, timesteps, dropout masks and the like once, outside, and pass
  them in `inputs` so all three runs see the same values.

## A whole model through a booster plugin

`tests.kit.numerics.scenarios.independent_impl.check_training` trains a `tests/kit/model_zoo` entry for a few steps through a booster
plugin (`test`) and as plain PyTorch (`ref` at the plugin's precision, `truth` in float64), from the same weights,
batches and upstream gradients. An entry takes part once its `ModelAttribute` carries a `NumericsSpec`; for a
HuggingFace model nothing else is needed:

```python
model_zoo.register(
    name="transformers_mixtral",
    model_fn=lambda: transformers.MixtralModel(config),
    ...,
    model_attribute=ModelAttribute(has_control_flow=True, numerics=NumericsSpec()),
)
```

Pass the entry's model_zoo name and the plugin exactly as training code would build it. Plugins need
`torch.distributed`, so run inside `colossalai.testing.spawn`:

```python
import colossalai
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.testing import spawn
from tests.kit.numerics import judge
from tests.kit.numerics.scenarios.independent_impl import blank_control, training_traces


def run(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    blank_control("transformers_mixtral")
    plugin = LowLevelZeroPlugin(stage=2, precision="bf16")
    test, ref, truth = training_traces("transformers_mixtral", plugin, steps=3)
    print(judge(test, ref, truth))


if __name__ == "__main__":
    spawn(run, 1)
```

The traces hold, for the first step, the outputs `out/...`, input and parameter gradients `grad/...` and the weights
after the optimizer step `param/...` (AdamW, lr 1e-3, on both sides), and `loss/<step>` for every step, so `steps=N`
also compares a short loss curve step by step. The loss is a fixed random projection of all outputs, so every output
receives a gradient.

### Keeping the precision recipes equal

A comparison is only meaningful if `ref` reduces precision exactly where the plugin does. Three checks enforce this:

1. **The recipe follows from the configuration.** `capture.recipes.recipe_for(plugin, mixed_precision)` derives how the
   reference runs; anything not checked against the plugin source raises `UnsupportedRecipe`:

   | Configuration | Reference | Device |
   | --- | --- | --- |
   | `TorchDDPPlugin()` | float32 | CPU or GPU |
   | `TorchDDPPlugin()`, `mixed_precision="fp16"` | float32 weights, `torch.autocast(float16)`, loss scale 2**16 | CUDA |
   | `TorchDDPPlugin()`, `mixed_precision="bf16"` | refused: `BF16MixedPrecision` is not implemented | |
   | `LowLevelZeroPlugin(precision="fp32")`, stage 1 or 2 | float32 | CUDA or NPU |
   | `LowLevelZeroPlugin(precision="bf16")`, stage 1 or 2 | whole model and inputs cast to bf16, float32 master weights | CUDA or NPU |
   | `HybridParallelPlugin(tp_size=1, pp_size=1, precision="fp32")` (the shardformer path) | float32 | CUDA or NPU |
   | `HybridParallelPlugin(tp_size=1, pp_size=1, precision="bf16")` | whole model and inputs cast to bf16, float32 master weights | CUDA or NPU |
   | `LowLevelZeroPlugin` with fp16, `max_norm`, `use_fp8`, `cpu_offload` or `cast_inputs=False`; `HybridParallelPlugin` with any parallel size above 1, sequence parallelism, `zero_stage > 0`, `max_norm`, `use_fp8`, `cpu_offload` or fp16; other plugins | refused: not verified yet | |

2. **The dtypes actually used must match.** Both runs record the input, output and weight dtypes of every module,
   the gradient dtypes and the optimizer state dtypes. Any difference raises `RecipeMismatch` before numbers are
   compared: with an autocast plugin and a whole-cast reference, 68 records differ for Mixtral, while judging the
   numbers would have reported 40 of 44 correct tensors as failures.
3. **Blank control.** `blank_control(entry)` runs a configuration in which the plugin changes no arithmetic (one
   device, `LowLevelZeroPlugin(stage=1, precision="bf16")`, or `TorchDDPPlugin()` on CPU) and requires every ratio to
   be 1. Run it first for a new entry: it catches a wrong reference implementation, before any real comparison.

The reference is also run twice. Tensors that differ between the two runs come from nondeterministic kernels; they
are listed in `ref.meta["nondeterministic"]`, and the worse of the two runs is used as their baseline.

On a single device every supported configuration reproduces the reference exactly (all ratios 1.00 for Mixtral,
Llama, GPT and BERT). Ratios above 1 appear once a plugin changes the arithmetic: more devices, replaced kernels,
other parallel layouts.

### What an entry has to satisfy

- **Runs in float64.** Its config must choose implementations that support float64: `attn_implementation="eager"`,
  and for HuggingFace MoE models `experts_implementation="eager"` (the default `grouped_mm` kernel has no float64).
- **No randomness inside `forward`.** Models run in train mode with every dropout probability set to 0, so that
  training-only paths (auxiliary losses, routers) are exercised. Noise, timesteps and masks must be generated in
  `data_gen_fn`, which is called once per step.
- **Other references.** If the reference is not the entry's own model (e.g. an official repository), pass
  `NumericsSpec(reference_fn=..., param_map=...)`: `reference_fn` builds it with the same inputs and output structure,
  `param_map` maps its parameter names to the model's. `output_fn` selects which outputs are compared.
- **Unused parameters.** Trainable parameters that the entry's inputs never reach must be listed in
  `NumericsSpec(unused_params=...)`; any other parameter left without a gradient fails.

What a check can detect depends on the tensor. In the self-tests a dropped column of a bf16 MLP moves the outputs,
the affected gradients and every step's loss 17 to 26 times beyond the baseline error, but not the weights after one
AdamW step: the first update is about `lr * sign(grad)`, which hardly depends on the size of a gradient error.

These checks are run by hand while developing a model and attached to its review; CI runs only the tool's own tests
in `tests/test_numerics`.

### Mixture of experts: routing flips

At low precision a router can send a token to a different set of experts than the float64 truth does. That token
then gets other experts' outputs, so the tensors downstream of it can exceed the baseline error by far although
nothing is wrong. Declare the routers and the report counts these flips:

```python
spec = NumericsSpec(routers=Routers(select=lambda name, module: name.endswith("mlp.gate"), top_k=8))
```

The `transformers_mixtral` entry declares its routers this way (`tests/kit/model_zoo/transformers/mixtral.py`); use a
module-level function rather than a lambda where the entry is passed to spawned processes.

`select(name, module)` picks the router modules by qualified name, in whichever model is being run; calls are matched
between runs by their order, so a reference with other names works if `select` matches its routers too. Each run
records, for its first forward, the sorted top-`top_k` experts of every token at every router call
(`meta["routing"]`, saved with the trace). `scores_fn` picks the scores from the router's output (default: the output,
or its first element for a tuple); a model that adds a bias or limits expert groups before its top-k must return
those adjusted scores.

The report then holds `routing`: the number of token routings, the flips against the truth in `test` and in `ref`,
and the same per router call. Flips in `ref` are what the precision itself causes; a `test` with many more is
suspect. Failing tensors get `routing: N flips in test, M in ref` appended to their reason. The counts never change a
verdict; a router missing from a run, or calls that route different numbers of tokens, fail as `<routing>`.
For a mixture-of-experts layer on its own, pass the same `routers=` to `check_fwd_bwd`.

## Data-parallel equivalence

`scenarios.data_parallel.check` checks that N-way data parallelism computes what one device computes when it accumulates
the same N micro-batches. Both sides run the same plugin at the same precision, so there is no reference
implementation and no float64 truth; the rule is Megatron-LM's relative bound instead
(`test_optimizer_grads_match.py`): per gradient, `||test - ref|| / ||ref|| <= k * C**depth * eps`, with Megatron's
constants `k = 4`, `C = 1.03`, depth 33 and the bfloat16 epsilon, about 8.3%.

Call it from the main process; it spawns a one-device run and a `world_size`-device run, saves both traces and
compares them. The plugin is passed as a function building it, since each process builds its own:

```python
from functools import partial

from colossalai.booster.plugin import LowLevelZeroPlugin
from tests.kit.numerics.scenarios import data_parallel

report = data_parallel.check(
    "transformers_mixtral", partial(LowLevelZeroPlugin, stage=2, precision="bf16"), world_size=2, micro_batches=4
)
report.assert_pass()
```

The report has the same columns as `judge`, with the bound in place of the baseline error, so `ratio` must be at most
1. Measured on Mixtral, Llama, GPT and BERT with 2 devices and 4 micro-batches: at most 5% of the bound in bf16 and
about 1e-7 relative error in fp32; a parameter left out of gradient synchronization is off by 99.8%, 12 times the
bound.

As in Megatron-LM only the gradients of one step are judged. The traces also hold the weights after the AdamW step
and the micro-batch losses, but neither is a stable target: the first Adam update is about `lr * sign(grad)`, so a
rounding difference in a near-zero gradient flips that element's update (a GPT-2 bias was 365 times beyond the
float32 bound after one step while every gradient was within it), and the random-projection loss mostly cancels,
which inflates its relative error.

## Resuming from a checkpoint

`scenarios.resume.check` checks that training resumed from a checkpoint is identical to training that was never
interrupted. It spawns three runs of the same plugin: `steps_before + steps_after` steps without interruption,
`steps_before` steps that save a checkpoint with the booster (model sharded in safetensors with its HuggingFace
config, optimizer, learning rate scheduler), and a fresh process group that builds everything anew, loads the
checkpoint and runs the remaining steps. Every step is deterministic, so the rule is exact equality
(`judge_equal`, Megatron-LM's `DeterministicTest`): the losses of the resumed steps and the final weights must be the
same bits. The learning rate warms up over the whole run, so a schedule that was not restored shows too. The saved
model must also load with HuggingFace `from_pretrained` and give exactly the weights it was saved with.

```python
from functools import partial

from colossalai.booster.plugin import LowLevelZeroPlugin
from tests.kit.numerics.scenarios import resume

report = resume.check("transformers_mixtral", partial(LowLevelZeroPlugin, stage=2, precision="fp32"), world_size=2)
report.assert_pass()
```

`parts=("model", "optimizer")` and the like leave a part out of the checkpoint; each omission is detected. Results on
Mixtral, Llama, GPT and BERT with 1 and 2 devices:

| Configuration | Resumed run |
| --- | --- |
| `TorchDDPPlugin()` fp32, `LowLevelZeroPlugin(precision="fp32")` stage 1 and 2 | identical; `from_pretrained` reads every weight back |
| `TorchDDPPlugin()` with `mixed_precision="fp16"` | identical with `shard_optimizer=False`; a sharded optimizer fails to load (see below) |
| `LowLevelZeroPlugin(precision="bf16")` stage 1 and 2 | **not identical**: weights differ in the last bf16 bit (e.g. 2**-8 around 1) |
| `HybridParallelPlugin(tp_size=1, pp_size=1)` fp32 (Mixtral, 1 device) | identical |
| `HybridParallelPlugin(tp_size=1, pp_size=1, precision="bf16")` (Mixtral, 1 device) | **not identical**, the same symptom as LowLevelZero bf16; the cause is not yet traced |

The bf16 difference is real: LowLevelZero does not save the float32 master weights. On load,
`update_master_params` rebuilds them from the bf16 weights, so their low bits are lost and later updates round
differently. Megatron-LM, for comparison, checkpoints the float32 parameters with the optimizer state.

Two loading failures, not specific to this tool, are worked around with `shard_optimizer=False`:

- On CPU, loading a sharded optimizer raises: `load_states_into_optimizer` calls `get_accelerator().synchronize()`,
  which the CPU accelerator does not support. The CPU self-tests save the optimizer unsharded.
- With `TorchDDPPlugin` and `mixed_precision="fp16"`, loading a sharded optimizer raises
  `'OptimizerWrapper' object has no attribute 'state'`: the optimizer is wrapped twice and only one wrapper is removed.

## L2 acceptance of a model line

Each model line runs the acceptance of its own model and hands in the report; the tool's owners review it. The rules
are the tool's defaults and cannot be changed from the command line. If a check fails, or a rule looks wrong for a
model, open an issue with the report attached instead of loosening the rule on a branch.

The five numeric checks of L2, and what covers them:

| L2 check | Covered by |
| --- | --- |
| ① aligned with the reference: logits, loss, per-layer gradients, weights after one step | `independent_impl` of the acceptance command |
| ② N-device data parallel equals single-device gradient accumulation | `data_parallel` of the acceptance command |
| ③ every replaced operator (fused kernel, Triton, chunked implementation) matches a plain one | an operator test written by the line, below |
| ④ resuming from a checkpoint equals uninterrupted training; the HuggingFace export loads with `from_pretrained` | `resume` of the acceptance command |
| ⑤ short loss curve with the same seed and data order matches the reference | `independent_impl` (every step's loss is judged) |

### What a line writes

1. **Its model_zoo entry**, the one L1 needs anyway, with `ModelAttribute(numerics=NumericsSpec(...))` and the
   requirements of [What an entry has to satisfy](#what-an-entry-has-to-satisfy). `NumericsSpec()` is enough for a
   HuggingFace model.
2. **One operator test per replaced operator**, in the repository under `tests/`: a plain implementation written from
   the formula, compared with `check_fwd_bwd`, and `expect_calls` naming the fused function, so that a run that
   silently fell back to an unfused path fails:

   ```python
   report = check_fwd_bwd(FusedRMSNorm(...), rms_norm_from_formula, inputs, torch.bfloat16,
                          expect_calls=[(fused_module, "fused_rms_norm")])
   report.assert_pass()
   ```

   `expect_calls` takes `(owner, name)` pairs and counts the calls of `getattr(owner, name)` during the test run
   (`count_calls`); as with `unittest.mock.patch`, `owner` is where the calling code looks the name up.

Checks ② and ④ need nothing more: they compare two runs of ColossalAI's own code.

### Running it

```bash
python -m tests.kit.numerics.acceptance --entry transformers_xxx --out transformers_xxx.json
```

By default every check runs under every configuration: `ddp_fp32`, `ddp_fp16`, `ddp_bf16`, `zero1_fp32`,
`zero1_bf16`, `zero2_fp32`, `zero2_bf16`, `zero2_bf16_offload`, `shardformer_fp32`, `shardformer_bf16`
(`HybridParallelPlugin` with `tp_size=1`). `--configs` and `--checks` (comma-separated) select a subset while
developing; the report handed in covers them all. `data_parallel` needs two devices, the rest one. Run from a pushed
commit with a clean tree: the report records the commit and whether the tree was dirty, and a dirty report cannot be
reproduced. The command exits with 1 if anything failed.

### The report

One JSON file, plus a summary table printed to the terminal. Each result is one check under one configuration:

| Field | Content |
| --- | --- |
| `check`, `config`, `l2_item` | what ran, and the L2 check it answers |
| `status` | `passed`; `failed` (also a run that raised, with `reason` and `error`); `unsupported` (the tool refuses the configuration, e.g. an unverified precision recipe); `not_run` (this machine cannot run it, e.g. too few devices) |
| `rule`, `tensors`, `max_ratio` | the judging rule, how many tensors were compared, the largest ratio among those that passed |
| `failures` | per failing tensor: `name`, `reason`, `allowed`, `error`, `ratio` |
| `routing` | only for entries that declare `Routers`: token routings, flips in test and ref, per router call |

The top level holds `schema_version`, `entry`, `created`, `command`, `repository` (`commit`, `dirty`, `changed`),
`environment` (torch, transformers and ColossalAI versions, accelerator, device name and count), `steps` and
`summary` (count per status).

Reports go to `/mnt/beegfs/ColossalAI/shared-results/numerics/<line>/<entry>/<YYYYMMDD>-<commit[:8]>.json`, where
`<line>` is the roadmap's line number and `<entry>` the model_zoo name, e.g.
`shared-results/numerics/2A/transformers_qwen3_moe/20261105-2e6bb3c0.json`: a new file for each acceptance run, never
overwriting an older one. They are not kept in `shared-cache`, which holds model
weights in the HuggingFace hub layout with per-file checksums. Once model-level checks run in CI, their expected
values will move into the repository, as PyTorch and Megatron-LM keep theirs.

### Per line

| Line | Special case | What to do now |
| --- | --- | --- |
| 2A MoE | a router flip makes single tensors jump | declare `NumericsSpec(routers=...)` (and `routers=` of `check_fwd_bwd`); the report counts flips in test and ref |
| 2B hybrid architectures | multimodal model trained on its text path | frozen vision parameters need no gradient; list skipped trainable branches in `unused_params` |
| 2C, 2D diffusion | random masks, noise, timesteps | generate them in `data_gen_fn` |
| 2E optimizers | no reference model; the update rule is compared | the reference-optimizer entry point is not there yet |
| 2F video generation, 2G Cosmos 3 | outputs are not logits; the reference is not HuggingFace | `output_fn` selects the tensors, `reference_fn` builds the official implementation |

Supported plugins are `TorchDDPPlugin`, `LowLevelZeroPlugin` and `HybridParallelPlugin` with `tp_size=1`. Gemini,
FSDP, tensor, pipeline and expert parallelism are not supported yet and are reported as `unsupported`, which is not a
failure.

## Layout

Three layers, passing `Trace`s:

| Layer | Modules | Role |
| --- | --- | --- |
| Scenarios | `scenarios.independent_impl`, `data_parallel`, `resume` | which runs to make and which rule judges them, one module per question |
| Capture | `capture.fwd_bwd` (one forward and backward, torch only), `capture.training` and `capture.recipes` (training through a plugin, ColossalAI) | run once, return a `Trace` |
| Judge | `judge`, `judge_relative_bound`, `judge_equal` | compare traces, return an `AlignmentReport` |

Only `independent_impl` compares against another codebase; the other scenarios compare two runs of the same code that differ in one condition. Every layer can be used on its own. For an operator or module, `scenarios.independent_impl.check_fwd_bwd(test_fn, ref_fn, inputs, dtype)` does the three `run_fwd_bwd` runs and the `judge` of the examples above in one call.

## API

| Name | Purpose |
| --- | --- |
| `assert_aligned(test, ref, truth, tol=1e-4)` | judge and raise with the report if anything is not aligned |
| `judge(test, ref, truth, tol=1e-4, small_tensor_multiplier=False)` | return an `AlignmentReport` |
| `scenarios.independent_impl.check_fwd_bwd(test_fn, ref_fn, inputs, dtype, unused_params=(), expect_calls=(), routers=None)` | the three runs and the judge of an operator or module check; `expect_calls` must each be called by the test run |
| `count_calls(targets)` | context manager counting the calls of each `(owner, name)` |
| `run_fwd_bwd(fn, inputs, dtype, grad_seed=0, unused_params=(), **meta)` | run a function or module forward and backward, return a `Trace` |
| `NumericsSpec(reference_fn=None, param_map=None, output_fn=None, unused_params=(), routers=None)` | opts a model_zoo entry into whole-model alignment |
| `Routers(select, top_k, scores_fn=...)` | the router modules of a mixture-of-experts model, for routing-flip counts |
| `diagnose_routing(report, test, ref, truth)`, `RoutingRecorder(module, routers)` | count the flips recorded in `meta["routing"]`; the hook that records them |
| `scenarios.independent_impl.training_traces(entry, plugin, mixed_precision=None, steps=1)` | train an entry through a plugin and as the reference, return `(test, ref, truth)` |
| `scenarios.independent_impl.check_training(...)`, `scenarios.independent_impl.blank_control(entry)` | `judge` the above; check the reference where the plugin changes nothing |
| `scenarios.data_parallel.traces(entry, plugin_fn, mixed_precision=None, world_size=2, micro_batches=None)` | run on `world_size` devices and on one, return `(test, ref)` |
| `scenarios.data_parallel.check(...)` | `judge_relative_bound` on the gradients of the above |
| `scenarios.resume.traces(entry, plugin_fn, mixed_precision=None, world_size=1, steps_before=2, steps_after=2, parts=..., shard_optimizer=True)` | uninterrupted, saved and resumed runs |
| `scenarios.resume.check(...)` | `judge_equal` on the resumed steps, plus the HuggingFace round trip |
| `judge_equal(test, ref)` | bitwise equality, Megatron-LM's `DeterministicTest` |
| `judge_relative_bound(test, ref, dtype, k=4.0, C=1.03, depth=33)`, `relative_bound(dtype)` | Megatron-LM's rule for runs of the same code |
| `require_gradients(report, **traces)` | fail the parameters the runs recorded in `meta["missing_grads"]` |
| `capture.recipes.recipe_for(plugin, mixed_precision)` | the reference recipe for a configuration, or `UnsupportedRecipe` |
| `capture.training`: `prepare`, `run_reference`, `run_boosted`, `run_data_parallel`, `run_steps` | the individual runs, for new scenarios |
| `Trace`, `Trace.capture`, `Trace.save`, `Trace.load` | one run as named tensors plus metadata |
| `error_multiplier(dtype, numel)` | the multiplier used for a tensor |
| `tensor_metrics(actual, expected)` | RMSE, max abs, relative L2 and cosine, in float64 |
| `acceptance.run_acceptance(entry, configs, checks, steps=3)`, `acceptance.format_report(report)` | the L2 acceptance report as a dict, and its summary table; `python -m tests.kit.numerics.acceptance` writes it to JSON |
| `cast_floating`, `full_precision_matmul`, `flatten_tensors`, `missing_grads`, `upstream_loss` | helpers used by the above |
