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
also carries `metrics` with max abs error, relative L2 error and cosine similarity.

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

## API

| Name | Purpose |
| --- | --- |
| `assert_aligned(test, ref, truth, tol=1e-4)` | judge and raise with the report if anything is not aligned |
| `judge(test, ref, truth, tol=1e-4, small_tensor_multiplier=False)` | return an `AlignmentReport` |
| `run_fwd_bwd(fn, inputs, dtype, grad_seed=0, **meta)` | run a function or module forward and backward, return a `Trace` |
| `Trace`, `Trace.capture`, `Trace.save`, `Trace.load` | one run as named tensors plus metadata |
| `error_multiplier(dtype, numel)` | the multiplier used for a tensor |
| `tensor_metrics(actual, expected)` | RMSE, max abs, relative L2 and cosine, in float64 |
| `cast_floating`, `full_precision_matmul`, `flatten_tensors` | helpers used by the above |

Whole-model alignment that builds models from `tests/kit/model_zoo` and runs them through booster plugins is not part
of this module yet.
