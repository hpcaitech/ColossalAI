"""Deterministic tests for the dataloader worker seeding of ``HybridParallelPlugin.prepare_dataloader``.

Why the seeding function has to live at module scope
----------------------------------------------------
``prepare_dataloader`` gives every worker the same starting seed through ``worker_init_fn``. That function used to be
a closure defined inside the method, which works with the default ``fork`` start method -- a forked worker inherits
the parent's memory, so nothing is ever pickled. With ``"spawn"`` (and ``"forkserver"``) torch pickles everything the
worker needs -- the dataset, ``collate_fn`` and ``worker_init_fn`` -- and sends it to a fresh interpreter. A function
defined inside another function can only be pickled by reference to a qualified name such as
``HybridParallelPlugin.prepare_dataloader.<locals>.seed_worker``, which does not exist at module scope in the child,
so pickling fails with::

    AttributeError: Can't pickle local object
    'HybridParallelPlugin.prepare_dataloader.<locals>.seed_worker'

and the loader raises before it yields its first batch. The seeding function now lives at module scope and receives
its seed through :func:`functools.partial`; ``partial`` pickles by reference to that importable function plus its
keyword arguments, which every start method can reconstruct.

What is checked here
--------------------
* the ``worker_init_fn`` of a spawn-configured loader pickles, and still refers to the module-level function -- the
  regression itself;
* a spawn-configured loader actually starts its workers and yields every sample;
* the seeding is unchanged: each worker's ``numpy``/``torch``/``random`` streams start exactly where a fresh
  ``seed`` puts them, and every worker sees the same stream, which is what the closure did;
* ``fork`` workers, which never needed pickling, produce the same stream;
* a loader with ``num_workers=0``, which never calls ``worker_init_fn``, is unaffected.

The tests only need a single-process ``gloo`` group: the plugin's ``prepare_dataloader`` reads the DP rank and builds
a sampler, and nothing else. No accelerator is touched.
"""

import functools
import multiprocessing
import pickle
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelPlugin, _seed_data_loader_worker
from colossalai.testing import free_port

SEED = 1024


class _DrawDataset(Dataset):
    """A picklable dataset whose items carry the worker's first ``torch.rand`` draw.

    ``worker_init_fn`` runs in the worker before the first ``__getitem__``, so with the plugin's seeding every worker
    starts from the same stream and draws the same value first. A worker seeded by torch alone (``base_seed +
    worker_id``) would draw a different value, which makes the draw a direct observation of the seeding.
    """

    def __init__(self, num_samples: int = 2) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        return index, torch.rand(1).item()


def _expected_draws(seed: int):
    """The first draw of each RNG after seeding it with ``seed``, i.e. what ``_seed_data_loader_worker`` sets up."""
    np.random.seed(seed)
    numpy_draw = float(np.random.rand())
    torch.manual_seed(seed)
    torch_draw = float(torch.rand(1).item())
    random.seed(seed)
    python_draw = random.random()
    return numpy_draw, torch_draw, python_draw


@pytest.fixture(scope="module")
def plugin():
    """A one-rank plugin; ``prepare_dataloader`` only needs the DP group and its rank."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{free_port()}",
        rank=0,
        world_size=1,
    )
    try:
        yield HybridParallelPlugin(tp_size=1, pp_size=1, zero_stage=0, precision="fp32")
    finally:
        dist.destroy_process_group()


def test_spawn_worker_init_fn_is_picklable_and_module_level(plugin):
    loader = plugin.prepare_dataloader(_DrawDataset(), batch_size=1, num_workers=2, multiprocessing_context="spawn")

    # This is what torch does to every argument it hands a spawn worker. A closure defined inside
    # `prepare_dataloader` raised AttributeError here, before a single worker was started.
    restored = pickle.loads(pickle.dumps(loader.worker_init_fn))

    assert isinstance(restored, functools.partial)
    assert restored.func is _seed_data_loader_worker
    assert restored.keywords == {"seed": SEED}


def test_spawn_workers_start_and_yield_every_sample(plugin):
    # One sample per worker, so each worker draws exactly once; all of them must draw the seeded first value.
    loader = plugin.prepare_dataloader(
        _DrawDataset(2), batch_size=1, num_workers=2, multiprocessing_context="spawn", persistent_workers=False
    )

    indices, draws = [], []
    for index, draw in loader:
        indices.append(int(index))
        draws.append(float(draw))

    assert sorted(indices) == [0, 1]
    assert sorted(draws) == pytest.approx([_expected_draws(SEED)[1]] * 2)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork is not available")
def test_fork_workers_produce_the_same_seeded_stream(plugin):
    # fork never pickles the worker arguments, so this path never broke; it must keep behaving exactly the same.
    loader = plugin.prepare_dataloader(
        _DrawDataset(2), batch_size=1, num_workers=2, multiprocessing_context="fork", persistent_workers=False
    )

    draws = sorted(float(draw) for _, draw in loader)

    assert draws == pytest.approx([_expected_draws(SEED)[1]] * 2)


def test_worker_init_fn_seeds_numpy_torch_and_random(plugin):
    # The seeding itself is unchanged: same three RNGs, same seed, for every worker id.
    loader = plugin.prepare_dataloader(_DrawDataset(), batch_size=1, num_workers=2, multiprocessing_context="spawn")
    worker_init_fn = pickle.loads(pickle.dumps(loader.worker_init_fn))
    expected = _expected_draws(SEED)

    for worker_id in (0, 1, 7):
        worker_init_fn(worker_id)
        assert (float(np.random.rand()), float(torch.rand(1).item()), random.random()) == pytest.approx(expected)


def test_loader_without_workers_is_unaffected(plugin):
    # Without workers torch never calls `worker_init_fn`; the loader must keep yielding every sample.
    loader = plugin.prepare_dataloader(_DrawDataset(2), batch_size=2, num_workers=0)

    assert loader.num_workers == 0
    index, _ = next(iter(loader))
    assert index.tolist() == [0, 1]
