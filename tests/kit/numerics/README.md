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
scale          = RMS(truth)
pass if  test error <= multiplier * baseline error + tol / 10 * scale
```

`multiplier` is 3 for fp16 / bf16 and 2 for fp32, and at least 3 for tensors with fewer than 1000 elements. A `test`
that is already `allclose` to `ref` (`atol = tol * scale`, `rtol = tol`, default `tol = 1e-4`) passes directly. This
is the rule PyTorch uses to check `torch.compile` across its model suites (`same()` in `torch/_dynamo/utils.py`),
with one change: PyTorch's `tol` is absolute, ours is relative to the size of each tensor. PyTorch compares every
parameter gradient too, with `tol = 1e-2` in training, and accepts that an absolute tolerance exceeds the whole
error of small tensors (gradients of a mean loss are often around 1e-6): its suites look for gross compiler errors
across hundreds of models. Here it would let any bug in such a tensor pass, so the `tol` terms are scaled and only
matter where the baseline error is about 0; multiplying a tensor by a constant does not change its verdict.

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
   | `TorchDDPPlugin()`, `mixed_precision="bf16"` or `BF16MixedPrecision()` | float32 weights, `torch.autocast(bfloat16)`, float32 optimizer states, no loss scaling | CUDA or NPU |
   | `LowLevelZeroPlugin(precision="fp32")`, stage 1 or 2 | float32 | CUDA or NPU |
   | `LowLevelZeroPlugin(precision="bf16")`, stage 1 or 2 | whole model and inputs cast to bf16, float32 master weights | CUDA or NPU |
   | `HybridParallelPlugin(tp_size=1, pp_size=1, precision="fp32")` (the shardformer path) | float32 | CUDA or NPU |
   | `HybridParallelPlugin(tp_size=1, pp_size=1, precision="bf16")` | whole model and inputs cast to bf16, float32 master weights | CUDA or NPU |
   | `LowLevelZeroPlugin(precision="bf16", cpu_offload=True)` | as bf16, with the master weights and the optimizer step on the CPU | CUDA or NPU |
   | `GeminiPlugin(precision="bf16")` | whole model and inputs cast to bf16, float32 master weights | CUDA |
   | `GeminiPlugin(precision="fp16")` | whole model and inputs cast to fp16, float32 master weights, loss scale `initial_scale` (2**16), divided out in float32 | CUDA |
   | `TorchFSDPPlugin()` | float32 | CUDA |
   | `TorchFSDPPlugin(mixed_precision=MixedPrecision(param_dtype=bf16 or fp16, buffer_dtype=the same))` | whole model and inputs cast, float32 master weights (FSDP's sharded parameters, which start from the model's own float32 weights, not from their cast), no loss scaling | CUDA |
   | `LowLevelZeroPlugin` with fp16, `max_norm`, `use_fp8`, `cast_inputs=False`, or `cpu_offload` with fp32 or `master_weights=False`; `HybridParallelPlugin` with any parallel size above 1, sequence parallelism, `zero_stage > 0`, `max_norm`, `use_fp8`, `cpu_offload` or fp16; `GeminiPlugin` with offloading (`placement_policy="auto"`, `offload_*_frac`), `max_norm`, `use_fp8`, `master_weights=False`, `tp_size` or `extra_dp_size` above 1; `TorchFSDPPlugin` with `cpu_offload`, `auto_wrap_policy`, `ignored_modules`, `fp8_communication`, a `buffer_dtype` other than `param_dtype` or `keep_low_precision_grads`; other plugins | refused: not verified yet | |

   CPU `BF16MixedPrecision` is refused: `CPUAccelerator.autocast` does not implement BF16 autocast. The acceptance
   command records `ddp_bf16` as `not_run` on CPU rather than running a float32 model under a BF16 label.

2. **The dtypes actually used must match.** Both runs record the input, output and weight dtypes of every module,
   the gradient dtypes, the dtypes of what the optimizer updates (the master weights where there are any) and of its
   state, each with its device type (`float32@cpu`), so that master weights and Adam's moments offloaded on one side
   only are caught as well. Any difference raises `RecipeMismatch` before numbers are compared: with an autocast plugin and a whole-cast reference, 68 records differ for Mixtral, while judging the
   numbers would have reported 40 of 44 correct tensors as failures.
3. **Blank control.** `blank_control(entry)` runs a configuration in which the plugin changes no arithmetic (one
   device, `LowLevelZeroPlugin(stage=1, precision="bf16")`, or `TorchDDPPlugin()` on CPU) and requires every ratio to
   be 1. Run it first for a new entry: it catches a wrong reference implementation, before any real comparison.

The reference is also run twice. Tensors that differ between the two runs come from nondeterministic kernels; they
are listed in `ref.meta["nondeterministic"]`, and the worse of the two runs is used as their baseline.

### Plugins that keep their own gradients: Gemini and FSDP

TorchDDP, LowLevelZero and HybridParallel expose each parameter's full gradient after backward. Gemini and FSDP keep
gradients in their own sharded, flattened buffers, so the tool reads them from the optimizer instead, with
Megatron-LM's trick: with Adam's `beta1 = 0` the first moment after a step is exactly that step's gradient, and both
plugins can gather the full optimizer state (`capture.plugins`). Both sides then run AdamW with `beta1 = 0`, so they
still run the same optimizer; the gradient, computed in the plugin's dtype and stored in float32, is cast back
exactly. Weights after the step come from the plugins' full state dicts. Two translations are needed:

- Gemini only accepts ColossalAI's `HybridAdam`; the tool rebuilds the AdamW as `HybridAdam(adamw_mode=True)` with
  the same hyperparameters, while the reference keeps torch's AdamW, an independent implementation of the same
  update.
- `TorchFSDPPlugin` rebuilds the optimizer with `optimizer.__init__(params, **optimizer.defaults)`. In torch 2.13
  AdamW's defaults include `decoupled_weight_decay`, which `AdamW.__init__` rejects, so FSDP with
  `torch.optim.AdamW` fails in ColossalAI itself; the tool passes `torch.optim.Adam(decoupled_weight_decay=True)`,
  the same update.

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
- **Every trainable parameter in the optimizer.** Each run checks, by identity, that the optimizer holds every
  parameter with `requires_grad=True`. An `optimizer_fn` that builds its own parameter groups and leaves one out does
  so on both sides, so every comparison would pass; it fails as `optim/param/<name>` instead.
- **Tied parameters.** Parameters that must be one `Parameter`, such as tied input and output embeddings, are declared
  with `NumericsSpec(tied_params=(("embed_tokens.weight", "lm_head.weight"),))`. Each run checks within its own model,
  and again after the plugin has wrapped it (FSDP excepted: its module holds plain tensors then), that every declared
  group is one Parameter and that no other names share one; a broken or undeclared tie fails as `tied/<names>`. A
  tied parameter is compared once, under its first name in `named_parameters()`.
- **Preparing the model.** `NumericsSpec(prepare_fn=...)` is called on every freshly built model, the one under test
  and the reference, before any run, e.g. to freeze a vision tower for a text-only training path
  (`model.visual.requires_grad_(False)`) or to replace a branch the training path skips. It may change the module in
  place or return a replacement. Frozen parameters need no gradient and keep their weights.
- **Gradient checkpointing.** Enable it in `prepare_fn`, so that every run recomputes the same way, and non-reentrant:
  `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` for a HuggingFace
  model. The self-tests check that each layer really runs again in backward, and that the result aligns, through
  `TorchDDPPlugin`, `LowLevelZeroPlugin` (bf16 with offload), `GeminiPlugin`, `TorchFSDPPlugin` and
  `HybridParallelPlugin` (bf16, `tp_size=1`). The reentrant variant leaves parameters without gradients when no input
  of a checkpointed block requires one, which then fails as a missing gradient.

What a check can detect depends on the tensor. In the self-tests a dropped column of a bf16 MLP moves the outputs,
the affected gradients and every step's loss 17 to 26 times beyond the baseline error, but not the weights after one
AdamW step: the first update is about `lr * sign(grad)`, which hardly depends on the size of a gradient error.

These checks are run by hand while developing a model and attached to its review; CI runs only the tool's own tests
in `tests/test_numerics`.

### Large models: one model on the device at a time

`prepare` keeps the model (and the reference) on the CPU. Each run copies the one it needs to the device and drops it
when it returns, so the runs never hold two models on the device together, and the float64 truth is cast on the way
to the device rather than copied there in float32 first. A run's peak is its own model, gradients and optimizer
state (16 bytes per parameter with AdamW in float32, 32 for the float64 truth), plus activations and the optimizer's
temporaries. A Llama with 111M parameters peaks at 5.05 GiB for the truth, 2.83 GiB for `ref` and 3.03 GiB through
`LowLevelZeroPlugin(stage=2, precision="bf16")` on one device. A model whose float64 truth does not fit is aligned
with a reduced configuration (fewer layers, same layer types); say so wherever the result is reported, as it does
not cover the full model.

### Real weights: caching the reference

Small random configs are cheap in float64. With real weights (`model_fn` loading them with `from_pretrained`, offline
from the shared cache) the float64 truth is the slow part, so `training_traces(..., cache_dir=...)` and
`acceptance --cache-dir` keep the reference runs and reuse them (`capture.cache`). A cached run is reused only if
nothing that determines it changed: the key hashes the entry name, seeds and number of steps, the reference weights
and every batch byte for byte, the code that selects outputs and routers, the optimizer's class and hyperparameters,
the recipe (for `ref` only: one float64 truth serves every configuration of an entry), the torch and transformers
versions, the device, and the source of the capture code itself, so a change to how references run invalidates old
entries without anyone bumping a version. Files are `<cache_dir>/<entry>/<truth|ref>-<key>.safetensors`, written
atomically, never overwritten and group-writable; the model under test always runs. Point a team at one shared
directory and a single float64 truth serves everyone.

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

Gemini and FSDP gradients are read from the optimizer here too (see above). Gemini accumulates micro-batches only
with `GeminiPlugin(enable_gradient_accumulation=True)`; without it a second backward overwrites the gradients, so the
check refuses that combination. With it, accumulation is wrong on more than one device: with 2 devices and 2
micro-batches each, Gemini's gradient is off by 69% in a minimal script (75 to 79% in the self-test MLP, and every
Mixtral gradient beyond the bound), while one device accumulating 4 micro-batches, and 2 devices with one micro-batch
each, equal plain PyTorch bit for bit.
`tests/test_zero/test_gemini/test_grad_accum.py` would cover this but is skipped without NVIDIA Apex. The acceptance
command reports it as a failure of `data_parallel`; the tool's own test uses one micro-batch per device.

By default the loss is the tool's own random projection of the outputs over equal micro-batches, which tests the
plugin's gradient reduction and nothing else. Models with padded batches have a second question: a training loop must
divide the summed loss by the valid tokens of the whole step (all micro-batches on all ranks), not average per
micro-batch, or micro-batches with fewer valid tokens weigh more. Declare the model's own loss and the check runs both
sides that way:

```python
spec = NumericsSpec(
    token_loss=TokenLoss(
        loss=lambda out, batch: F.cross_entropy(out.logits[:, :-1].flatten(0, 1), batch["labels"][:, 1:].flatten(),
                                                ignore_index=-100, reduction="sum"),
        count=lambda batch: (batch["labels"][:, 1:] != -100).sum(),
    )
)
report = data_parallel.check(entry, plugin_fn, world_size=2, micro_batches=4, token_weighted=True)
```

`count` reads the batch alone, as a training loop counts valid labels before the forward; every rank holds every
batch, so the step's total needs no communication. Each micro-batch's summed loss is scaled by
`world_size / total`, so N devices and one device both compute the gradient of the mean over all valid tokens.
Compared with DDP's own averaging over ranks (each rank's gradient divided by `world_size`), the factor `world_size`
cancels it, leaving one division by the step's valid tokens.

A micro-batch without any valid token (a sequence of padding only) still runs forward and backward: its gradient is
zero, the rank that holds it takes part in the reduction as usual, and its recorded loss is its sum, 0, not `0 / 0`.
A step without any valid token at all has no defined mean and raises `ValueError` on every rank before the first
forward, so no rank is left waiting in a collective.

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
same bits. The learning rate warms up over the whole run, so a schedule that was not restored shows too. The state
the checkpoint must carry is also compared directly after the last step, as the plugin holds it (its shards, flat
parameters or chunks, the same in both runs of one configuration): the tensors the optimizer updates
(`optim/<group>.<index>/param`, the float32 master weights where the plugin keeps them), Adam's moments and step,
the scheduler's state (`scheduler/...`) and the fp16 loss scale. A lost part then shows by name even where the
following steps happen to hide it. The saved model must also load with HuggingFace `from_pretrained` and give exactly
the weights it was saved with.

What the scenario carries itself, and a training loop must too, since a booster checkpoint does not:

- **Data progress.** The resumed run is told to start at batch `steps_before`; the checkpoint holds no dataloader
  position.
- **Random number generators.** Every run seeds them with the same seed when training starts and draws nothing while
  training (dropout is off; random inputs come from `data_gen_fn`). The generator states at the end (`rng/cpu`,
  `rng/cuda`) are compared too, so a model that does draw random numbers in training fails there: resuming it exactly
  needs the loop to save and restore the generator states.

```python
from functools import partial

from colossalai.booster.plugin import LowLevelZeroPlugin
from tests.kit.numerics.scenarios import resume

report = resume.check("transformers_mixtral", partial(LowLevelZeroPlugin, stage=2, precision="fp32"), world_size=2)
report.assert_pass()
```

`parts=("model", "optimizer")` and the like leave a part out of the checkpoint; each omission is detected.

The Gemma checkpoint integration adds FP32 master values to HybridParallel optimizer state, removes that private
metadata on load, and accepts legacy checkpoints without it (legacy checkpoints cannot recover discarded bits).
Gemini's gathered gradient accumulation also keeps the previous averaged gradient unscaled before the next
all-reduce; multiplying it by the world size counted earlier micro-batches multiple times. The sharded path still
compensates for its single-owner gradient shard. A linear model with hand-computed gradients tests both paths,
one/two devices and synchronous/asynchronous reduction without an Apex dependency.
It also exports the inner HF config for FSDP and the actual exported weight dtype for Gemini. DDP Torch AMP avoids
nesting the base optimizer wrapper and saves the scaler's complete state in private parameter-group metadata for
both sharded and unsharded checkpoints. Regression coverage is in `test_gemma4_checkpoint_regressions.py`.

For FP16 models needing a smaller initial loss scale, declare `NumericsSpec(fp16_loss_scale=1024)` in the entry.
The acceptance command applies it to DDP AMP and Gemini FP16, to both reference and plugin, and records it under
`recipe_overrides`. Other entries keep their defaults. Overflow, skipped steps and changed scales still fail;
neither the judge nor its tolerance is changed. Direct `check_training` callers must supply the same configured
mixed-precision instance or plugin themselves.

The following table records the original tool baseline, before those checkpoint fixes. Results on
Mixtral, Llama, GPT and BERT with 1 and 2 devices:

| Configuration | Resumed run |
| --- | --- |
| `TorchDDPPlugin()` fp32, `LowLevelZeroPlugin(precision="fp32")` stage 1 and 2 | identical; `from_pretrained` reads every weight back |
| `TorchDDPPlugin()` with `mixed_precision="fp16"` | identical with `shard_optimizer=False`; a sharded optimizer fails to load (see below) |
| `LowLevelZeroPlugin(precision="bf16")` stage 1 and 2 | **not identical**: weights differ in the last bf16 bit (e.g. 2**-8 around 1) |
| `HybridParallelPlugin(tp_size=1, pp_size=1)` fp32 (Mixtral, 1 device) | identical |
| `HybridParallelPlugin(tp_size=1, pp_size=1, precision="bf16")` (Mixtral, 1 device) | **not identical**, the same cause as LowLevelZero bf16: `HybridParallelCheckpointIO` rebuilds the master weights from the bf16 weights after loading |
| `LowLevelZeroPlugin(precision="bf16", cpu_offload=True)` (Mixtral, 1 device) | **not identical**, as LowLevelZero bf16 |
| `TorchFSDPPlugin()` fp32 and with bf16 `MixedPrecision` (Mixtral, 1 device) | identical (FSDP saves its float32 parameters); the HuggingFace export does not load, see below |
| `GeminiPlugin(precision="bf16")` (Mixtral, 1 device) | the sharded model checkpoint does not load, see below |

The bf16 difference is real: LowLevelZero does not save the float32 master weights. On load,
`update_master_params` rebuilds them from the bf16 weights, so their low bits are lost and later updates round
differently. Megatron-LM, for comparison, checkpoints the float32 parameters with the optimizer state.

Two loading failures, not specific to this tool, are worked around with `shard_optimizer=False`:

- On CPU, loading a sharded optimizer raises: `load_states_into_optimizer` calls `get_accelerator().synchronize()`,
  which the CPU accelerator does not support. The CPU self-tests save the optimizer unsharded.
- With `TorchDDPPlugin` and `mixed_precision="fp16"`, loading a sharded optimizer raises
  `'OptimizerWrapper' object has no attribute 'state'`: the optimizer is wrapped twice and only one wrapper is removed.

The unsharded workaround needs torch 2.10 or later: ColossalAI writes an unsharded optimizer with `torch.save` and
loads that format back only on torch 2.10 and later. On CPU with an older torch, `resume` therefore cannot run at all;
the acceptance command reports it as `not_run` and the self-tests skip it
(`scenarios.resume.unsharded_optimizer_reloads()` checks the installed torch).

Two more, which `resume` reports as failures:

- `GeminiPlugin`: loading a sharded model raises `GeminiDDP.named_parameters() got an unexpected keyword argument
  'remove_duplicate'`. `GeneralCheckpointIO.load_sharded_model` passes it when resolving tied parameters, and
  `GeminiDDP.named_parameters` does not accept it.
- `TorchFSDPPlugin`: a sharded model checkpoint has no `config.json`, so `from_pretrained` falls back to a default
  config and fails on the shapes. `save_config_file` is given `model.unwrap()`, which is the FSDP wrapper, not the
  HuggingFace model, and returns without writing.

## Calibrating the multiplier

`python -m tests.kit.numerics.calibrate` measures how the rule separates correct from buggy code: operators checked
with `check_fwd_bwd` against plain references written from the formula, three seeds each, on an H20. Largest ratio
over each case's outputs and gradients:

| Case | bf16 | fp16 |
| --- | --- | --- |
| correct: SDPA flash / efficient / math backends vs naive attention | 0.83-0.86 / 0.85-0.86 / 0.75-0.77 | 0.83-0.86 / 0.85-0.86 / 0.75-0.76 |
| correct: `F.rms_norm`, `F.layer_norm`, `F.cross_entropy` vs naive | 0.73-0.77, 1.00, 1.00 | 0.74-0.77, 1.00, 1.00 |
| correct: transformers' `grouped_mm` experts vs its eager loop | 1.00 | 1.00 |
| bug: causal mask off by one | 89-110 | 718-849 |
| bug: attention scale off by 1% | 2.9-3.1 | 22-23 |
| bug: RMSNorm / LayerNorm epsilon 1e-2 instead of 1e-6 / 1e-5 | 1.4-1.5 / 1.2-1.3 | 10 / 8.8 |
| bug: cross-entropy with label smoothing 0.01 | 2.7-2.8 | 17-18 |

Whole models through plugins on one device are exact (1.00) in every supported configuration but one, once the
reference follows the plugin in every detail: with master weights started from the bf16-cast weights instead of
FSDP's unrounded float32 ones, FSDP bf16 reached 1.42 on Mixtral and 237 on one loss of a model with tied embeddings.
With the fp16 loss scale divided out in fp16 instead of in float32 as Gemini does, Gemini fp16 reached 1.68 on
Mixtral. Both stayed below 3, so the self-tests require ratios of exactly 1 on one device. The exception is Gemini
fp16, now 1.07 on Mixtral: its fused `HybridAdam` and torch's `AdamW`, the independent reference, differ in the last
float32 bit of a few elements, which fp16's finer rounding sometimes shows (one element in 512 of a test model, one
fp16 unit, the same without loss scaling).

No correct implementation came near 3, so the multiplier stays. What bf16 cannot show is a bug smaller than its own
rounding: a 1% scale or an epsilon of 1e-2 changes results by about one bf16 unit, and lands at 1.2 to 3 whatever the
multiplier. The same bugs are 9 to 23 times beyond the baseline in fp16. Operator tests should therefore run in fp16
as well as bf16 (or fp32), not bf16 alone.

With PyTorch's absolute tolerance two of these bugs passed although their ratio exceeded 3: label smoothing in fp16
at 17 and the 1% scale in bf16 at 3.09 (one seed). `tol / 10 = 1e-5` alone was 60 times the allowance of 3 baseline
errors for the logits gradient, around 1e-6 per element. With the tolerance scaled to each tensor both fail, and
every correct case keeps its ratio and its pass.

## An optimizer's update rule

For a new optimizer (Muon, say) there is no reference model; what is compared is the update rule.
`scenarios.independent_impl.check_optimizer` runs the optimizer under test and a reference optimizer from the same
weights with the same gradients at every step, and judges the weights after each step with the baseline-error rule:
the reference in float64 is the truth, the reference at the tested precision the baseline.

```python
from tests.kit.numerics.scenarios.independent_impl import check_optimizer

report = check_optimizer(
    lambda params: Muon(params, lr=0.02),            # ColossalAI's implementation
    lambda params: ReferenceMuon(params, lr=0.02),   # e.g. the paper's official code, runs in float64
    model,                                           # a module (e.g. a model_zoo model) or a name -> tensor dict
    torch.bfloat16,
    steps=3,
)
report.assert_pass()
```

`params` gives the starting weights, copied for each run; the gradient of each parameter at step t is a standard normal
tensor seeded by `grad_seed + t` (produced in float64, then cast), or `grads_fn(step, name, weight)` for gradients of
a particular shape. Several steps exercise the optimizer state. The reference must match the tested one's internal
precision choices (an orthogonalization done in bfloat16, say), as the precision recipes do for whole models. Choose
a precision in which the updates are visible: in bfloat16 an update of `1e-3` is below the resolution of weights
around 1 and rounds away, so both a correct and a wrong rule leave the weights unchanged.

## Acceptance of a model

The acceptance command runs every check of a model under every supported configuration and writes one JSON report.
The rules are the tool's defaults and cannot be changed from the command line: if a check fails, or a rule looks
wrong for a model, report it with the JSON attached instead of loosening the rule on a branch.

The five numeric checks of a model's correctness review (level L2), numbered as in the report's `l2_item` field:

| Check | Covered by |
| --- | --- |
| ① aligned with the reference: logits, loss, per-layer gradients, weights after one step | `independent_impl` of the acceptance command |
| ② N-device data parallel equals single-device gradient accumulation | `data_parallel` of the acceptance command |
| ③ every replaced operator (fused kernel, Triton, chunked implementation) matches a plain one | an operator test written with the model, below |
| ④ resuming from a checkpoint equals uninterrupted training; the HuggingFace export loads with `from_pretrained` | `resume` of the acceptance command |
| ⑤ short loss curve with the same seed and data order matches the reference | `independent_impl` (every step's loss is judged) |

### What a model needs

1. **Its model_zoo entry**, the one the pass-through tests (L1) need anyway, with `ModelAttribute(numerics=NumericsSpec(...))` and the
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

Checks ② and ④ need nothing more: they compare two runs of ColossalAI's own code. For a model trained on padded
batches, also declare `NumericsSpec(token_loss=TokenLoss(...))` so that ② is checked with the model's own loss
normalized by valid tokens (`data_parallel_tokens`).

### Running it

```bash
python -m tests.kit.numerics.acceptance --entry transformers_xxx --out transformers_xxx.json
```

By default every check runs under every configuration: `ddp_fp32`, `ddp_fp16`, `ddp_bf16`, `zero1_fp32`,
`zero1_bf16`, `zero2_fp32`, `zero2_bf16`, `zero2_bf16_offload`, `shardformer_fp32`, `shardformer_bf16`
(`HybridParallelPlugin` with `tp_size=1`), `gemini_bf16`, `gemini_fp16` (with gradient accumulation enabled),
`fsdp_fp32`, `fsdp_bf16`. `--configs` and `--checks` (comma-separated) select a subset while developing; a report
meant as evidence covers them all. `data_parallel` (2 devices, 4 micro-batches) and `data_parallel_tokens` (the same with the
entry's `token_loss`; `not_run` for entries without one) need two devices, the rest one. With real weights, pass
`--cache-dir` to reuse the reference runs (see [Real weights](#real-weights-caching-the-reference)). Run from a pushed
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

Keep every report, one new file per run named by entry, date and commit (e.g.
`transformers_mixtral/20261105-2e6bb3c0.json`), so that two runs of the same model can be compared. Once
model-level checks run in CI, their expected values will move into the repository, as PyTorch and Megatron-LM keep
theirs.

### Special cases

| Model | Special case | What to do |
| --- | --- | --- |
| Mixture of experts | a router flip makes single tensors jump | declare `NumericsSpec(routers=...)` (and `routers=` of `check_fwd_bwd`); the report counts flips in test and ref |
| Multimodal model trained on its text path | the vision tower is frozen or skipped | freeze or replace it with `NumericsSpec(prepare_fn=...)`, applied to every run; list skipped trainable branches in `unused_params` |
| Tied embeddings | one Parameter under two names | declare them in `NumericsSpec(tied_params=...)`; compared once, under the first name |
| Gradient checkpointing | layers run again in backward | enable it in `prepare_fn`, non-reentrant: `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`, see [What an entry has to satisfy](#what-an-entry-has-to-satisfy) |
| Diffusion models | random masks, noise, timesteps | generate them in `data_gen_fn` |
| Optimizers | no reference model; the update rule is compared | `check_optimizer(test_fn, ref_fn, params, dtype, steps)` against the reference implementation, see [An optimizer's update rule](#an-optimizers-update-rule) |
| Video generation, other non-HuggingFace models | outputs are not logits; the reference is an official repository | `output_fn` selects the tensors, `reference_fn` builds the official implementation |

Supported plugins are `TorchDDPPlugin`, `LowLevelZeroPlugin`, `HybridParallelPlugin` with `tp_size=1`,
`GeminiPlugin` and `TorchFSDPPlugin`, in the configurations of the recipe table. Tensor, pipeline and expert
parallelism are not supported yet and are reported as `unsupported`, which is not a failure.

## Layout

Three layers, passing `Trace`s:

| Layer | Modules | Role |
| --- | --- | --- |
| Scenarios | `scenarios.independent_impl`, `data_parallel`, `resume` | which runs to make and which rule judges them, one module per question |
| Capture | `capture.fwd_bwd` (one forward and backward) and `capture.optimizer_step` (optimizer steps on given gradients), torch only; `capture.training`, `capture.recipes`, `capture.plugins` (training through a plugin) and `capture.cache` (reusing reference runs), ColossalAI | run once, return a `Trace` |
| Judge | `judge`, `judge_relative_bound`, `judge_equal` | compare traces, return an `AlignmentReport` |

Only `independent_impl` compares against another codebase; the other scenarios compare two runs of the same code that differ in one condition. Every layer can be used on its own. For an operator or module, `scenarios.independent_impl.check_fwd_bwd(test_fn, ref_fn, inputs, dtype)` does the three `run_fwd_bwd` runs and the `judge` of the examples above in one call.

## API

| Name | Purpose |
| --- | --- |
| `assert_aligned(test, ref, truth, tol=1e-4)` | judge and raise with the report if anything is not aligned |
| `judge(test, ref, truth, tol=1e-4, small_tensor_multiplier=False)` | return an `AlignmentReport`; `tol` is relative to each tensor's RMS in the truth |
| `scenarios.independent_impl.check_fwd_bwd(test_fn, ref_fn, inputs, dtype, unused_params=(), expect_calls=(), routers=None)` | the three runs and the judge of an operator or module check; `expect_calls` must each be called by the test run |
| `count_calls(targets)` | context manager counting the calls of each `(owner, name)` |
| `run_fwd_bwd(fn, inputs, dtype, grad_seed=0, unused_params=(), **meta)` | run a function or module forward and backward, return a `Trace` |
| `NumericsSpec(reference_fn=None, param_map=None, output_fn=None, unused_params=(), routers=None, prepare_fn=None, token_loss=None, tied_params=())` | opts a model_zoo entry into whole-model alignment |
| `TokenLoss(loss, count)` | a model's own loss summed over valid tokens, and the count of valid tokens, for `token_weighted` |
| `Routers(select, top_k, scores_fn=...)` | the router modules of a mixture-of-experts model, for routing-flip counts |
| `diagnose_routing(report, test, ref, truth)`, `RoutingRecorder(module, routers)` | count the flips recorded in `meta["routing"]`; the hook that records them |
| `scenarios.independent_impl.training_traces(entry, plugin, mixed_precision=None, steps=1, cache_dir=None)` | train an entry through a plugin and as the reference, return `(test, ref, truth)` |
| `scenarios.independent_impl.check_optimizer(test_fn, ref_fn, params, dtype, steps=1, grad_seed=0, grads_fn=None)` | an optimizer's update rule against a reference optimizer |
| `run_optimizer(optimizer_fn, params, dtype, steps=1, grad_seed=0, grads_fn=None)` | optimizer steps on copies of `params` with fixed gradients, return a `Trace` |
| `scenarios.independent_impl.check_training(...)`, `scenarios.independent_impl.blank_control(entry)` | `judge` the above; check the reference where the plugin changes nothing |
| `scenarios.data_parallel.traces(entry, plugin_fn, mixed_precision=None, world_size=2, micro_batches=None, token_weighted=False)` | run on `world_size` devices and on one, return `(test, ref)` |
| `scenarios.data_parallel.check(...)` | `judge_relative_bound` on the gradients of the above |
| `scenarios.resume.traces(entry, plugin_fn, mixed_precision=None, world_size=1, steps_before=2, steps_after=2, parts=..., shard_optimizer=True)` | uninterrupted, saved and resumed runs |
| `scenarios.resume.check(...)` | `judge_equal` on the resumed steps, plus the HuggingFace round trip |
| `judge_equal(test, ref)` | bitwise equality, Megatron-LM's `DeterministicTest` |
| `judge_relative_bound(test, ref, dtype, k=4.0, C=1.03, depth=33)`, `relative_bound(dtype)` | Megatron-LM's rule for runs of the same code |
| `require_gradients(report, **traces)` | fail the parameters the runs recorded in `meta["missing_grads"]` |
| `capture.recipes.recipe_for(plugin, mixed_precision)` | the reference recipe for a configuration, or `UnsupportedRecipe` |
| `capture.plugins`: `exposes_gradients`, `make_optimizer`, `full_gradients`, `full_parameters`, `FirstMomentIsGradient` | what differs between plugins: their optimizer, where the module is, reading gradients and weights back |
| `capture.cache`: `reference_key`, `cached_run` | the cache of reference runs |
| `capture.training`: `prepare`, `run_reference`, `run_boosted`, `run_data_parallel`, `run_steps` | the individual runs, for new scenarios |
| `Trace`, `Trace.capture`, `Trace.save`, `Trace.load` | one run as named tensors plus metadata |
| `error_multiplier(dtype, numel)` | the multiplier used for a tensor |
| `tensor_metrics(actual, expected)` | RMSE, max abs, relative L2 and cosine, in float64 |
| `acceptance.run_acceptance(entry, configs, checks, steps=3, cache_dir=None)`, `acceptance.format_report(report)` | the L2 acceptance report as a dict, and its summary table; `python -m tests.kit.numerics.acceptance` writes it to JSON |
| `calibrate.calibrate(dtypes, seeds)`, `python -m tests.kit.numerics.calibrate` | ratios of correct and buggy operator implementations |
| `cast_floating`, `full_precision_matmul`, `flatten_tensors`, `missing_grads`, `upstream_loss` | helpers used by the above |
