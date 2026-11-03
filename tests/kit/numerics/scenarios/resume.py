"""Scenario ④: training resumed from a checkpoint is identical to training that was never interrupted.

Three spawned runs of the same plugin: ``steps_before + steps_after`` steps without interruption; ``steps_before``
steps that save a checkpoint with the booster; and a fresh process group that builds everything anew, loads the
checkpoint and runs the remaining ``steps_after`` steps. The losses of the resumed steps and the final weights must
equal the uninterrupted ones bit for bit (``judge_equal``, Megatron-LM's ``DeterministicTest``): every step is
deterministic, so any difference is state the checkpoint did not carry, typically optimizer state, the learning rate
schedule or float32 master weights. That state is also compared directly at the end (``training.run_steps``): the
tensors the optimizer updates (``optim/<group>.<index>/param``, the float32 master weights where the plugin keeps
them), Adam's moments and step, the scheduler's state and the fp16 loss scale. A difference there shows which part
was lost even when the following steps hide it.

What the scenario carries itself, not the checkpoint (the boundary of what is tested):

- Data progress: the resumed run starts at batch ``steps_before`` because it is told so; a booster checkpoint holds
  no dataloader position. A training loop must save its own.
- Random number generators: a booster checkpoint does not save them. Every run seeds them with the same seed when
  training starts and the runs draw nothing while training (dropout is off; random inputs come from ``data_gen_fn``),
  which ``rng/cpu`` and ``rng/cuda`` (the generator states at the end) confirm: a model that draws random numbers in
  training fails there, and resuming it exactly needs a loop that saves and restores the generator states.

The checkpointed model must also load with HuggingFace ``from_pretrained`` (if it is a ``PreTrainedModel``) and give
exactly the weights it was saved with.
"""

import os
import tempfile
from typing import Callable, Iterable, Optional, Tuple, Union

import colossalai
from colossalai.booster.mixed_precision import MixedPrecision
from colossalai.testing import spawn

from ..capture.training import CHECKPOINT_PARTS, default_optimizer, prepare, run_steps
from ..judge import AlignmentReport, Entry, judge_equal
from ..trace import Trace


def traces(
    entry: Union[str, tuple],
    plugin_fn: Callable,
    mixed_precision: Union[str, MixedPrecision, None] = None,
    world_size: int = 1,
    steps_before: int = 2,
    steps_after: int = 2,
    parts: Iterable[str] = CHECKPOINT_PARTS,
    shard_optimizer: bool = True,
    seed: int = 0,
    grad_seed: int = 0,
    optimizer_fn: Callable = default_optimizer,
    directory: Optional[str] = None,
) -> Tuple[Trace, Trace, Trace, str]:
    """Return ``(resumed, uninterrupted, saved, checkpoint)``.

    ``saved`` is the run that wrote the checkpoint (its weights are what the checkpoint should hold) and
    ``checkpoint`` the checkpoint directory. Call from the main process; ``plugin_fn`` and ``entry`` must be
    picklable (see ``data_parallel.traces``). ``parts`` selects what is saved and loaded, to show what a missing part does.
    ``shard_optimizer=False`` saves the optimizer unsharded: loading a sharded optimizer fails on CPU, where
    ``load_states_into_optimizer`` calls the CPU accelerator's ``synchronize``, which raises.
    Without ``directory`` everything is written to a temporary directory that is deleted afterwards, so the returned
    checkpoint path is then only a name.
    """
    total = steps_before + steps_after
    kwargs = dict(
        entry=entry,
        plugin_fn=plugin_fn,
        mixed_precision=mixed_precision,
        total=total,
        seed=seed,
        grad_seed=grad_seed,
        optimizer_fn=optimizer_fn,
        parts=tuple(parts),
        shard_optimizer=shard_optimizer,
    )
    with tempfile.TemporaryDirectory() if directory is None else _keep(directory) as out:
        checkpoint = os.path.join(out, "checkpoint")
        paths = {role: os.path.join(out, f"{role}.safetensors") for role in ("uninterrupted", "saved", "resumed")}
        spawn(_run, world_size, role="uninterrupted", start=0, stop=total, path=paths["uninterrupted"], **kwargs)
        spawn(
            _run, world_size, role="saved", start=0, stop=steps_before, save=checkpoint, path=paths["saved"], **kwargs
        )
        spawn(
            _run,
            world_size,
            role="resumed",
            start=steps_before,
            stop=total,
            load=checkpoint,
            path=paths["resumed"],
            **kwargs,
        )
        traces = {role: Trace.load(path) for role, path in paths.items()}
        try:
            hf = _from_pretrained(entry, checkpoint)
        except Exception as exc:  # judged as one failed entry, so that the resumed run is still compared
            meta = dict(traces["saved"].meta, hf_error=f"{type(exc).__name__}: {exc}"[:500])
            traces["saved"] = Trace(dict(traces["saved"]), meta)
            hf = None
        if hf is not None:
            traces["saved"] = Trace(dict(traces["saved"], **hf), traces["saved"].meta)
        return traces["resumed"], traces["uninterrupted"], traces["saved"], checkpoint


def check(*args, **kwargs) -> AlignmentReport:
    """``judge_equal`` on the resumed steps and final weights, plus the HuggingFace round trip of the checkpoint."""
    resumed, uninterrupted, saved, _ = traces(*args, **kwargs)
    report = judge_equal(resumed, Trace({k: v for k, v in uninterrupted.items() if k in resumed}, {}))
    hf = {k[len("hf/") :]: v for k, v in saved.items() if k.startswith("hf/")}
    if "hf_error" in saved.meta:
        report.entries.append(Entry("hf/<from_pretrained>", False, f"from_pretrained failed: {saved.meta['hf_error']}"))
    elif hf:
        weights = {k: v for k, v in saved.items() if k.startswith("param/")}
        # every saved weight must come back; parameters HuggingFace adds on its own (e.g. a pooler the entry was built
        # without, initialized fresh by from_pretrained) are not part of the checkpoint and are ignored.
        # from_pretrained may load in another dtype; values saved in bf16 survive the cast back exactly
        loaded = {f"hf/{k}": v.to(weights[k].dtype) for k, v in hf.items() if k in weights}
        exported = judge_equal(loaded, {f"hf/{k}": v for k, v in weights.items()})
        report.entries.extend(exported.entries)
    else:
        report.entries.append(Entry("hf/<from_pretrained>", True, "not a PreTrainedModel, skipped", 0.0, 0.0, 0.0))
    return report


def unsharded_optimizer_reloads() -> bool:
    """Whether ColossalAI loads back an unsharded optimizer checkpoint it wrote, with the installed torch.

    It writes one with ``torch.save`` and loads that format only on torch 2.10 and later (``checkpoint_io/utils.py``).
    A sharded optimizer does not load on CPU (``load_states_into_optimizer`` synchronizes the CPU accelerator), so on
    CPU with an older torch ``check`` cannot run at all."""
    from colossalai.checkpoint_io.utils import load_optimizer_state, save_state_dict

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "optimizer.bin")
        save_state_dict({"state": {}, "param_groups": []}, path, use_safetensors=False)
        try:
            load_optimizer_state(path)
        except RuntimeError:
            return False
    return True


def _run(
    rank,
    world_size,
    port,
    entry,
    plugin_fn,
    mixed_precision,
    total,
    seed,
    grad_seed,
    optimizer_fn,
    parts,
    shard_optimizer,
    role,
    start,
    stop,
    path,
    load=None,
    save=None,
):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    s = prepare(entry, total, seed)
    trace = run_steps(
        s,
        plugin_fn(),
        start,
        stop,
        mixed_precision,
        optimizer_fn,
        grad_seed,
        load=load,
        save=save,
        parts=parts,
        shard_optimizer=shard_optimizer,
        role=role,
    )
    if rank == 0:
        trace.save(path)


def _from_pretrained(entry, checkpoint: str):
    # weights as HuggingFace reads the saved model back, keyed hf/param/<name> like the saved run's param/<name>
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return None
    if isinstance(entry, str):
        from tests.kit.model_zoo import model_zoo

        entry = model_zoo[entry]
    model_cls = type(entry[0]())
    if not issubclass(model_cls, PreTrainedModel):
        return None
    model = model_cls.from_pretrained(os.path.join(checkpoint, "model"))
    return {f"hf/param/{name}": p.detach().clone() for name, p in model.named_parameters()}


class _keep:
    def __init__(self, directory: str):
        self.directory = directory

    def __enter__(self) -> str:
        os.makedirs(self.directory, exist_ok=True)
        return self.directory

    def __exit__(self, *exc) -> None:
        pass
