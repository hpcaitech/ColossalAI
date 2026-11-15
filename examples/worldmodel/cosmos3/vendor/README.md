# Vendored DROID data pipeline

The 34 files under `cosmos_framework/` are copied from

    https://github.com/NVIDIA/cosmos-framework   commit 5950e7b4

**byte for byte**, with no edits of any kind -- no added provenance headers, no changed import paths, no
removed code. They are licensed under OpenMDW-1.1; the full text is in `LICENSE-OpenMDW-1.1`, copied from
the same commit.

## Why a copy instead of a reimplementation

This example used to read DROID through its own `droid_dataset.py`, written against the upstream behaviour.
That is a reimplementation, and a reimplementation can drift from what it reimplements without anything
failing. One such drift was measured: `droid_dataset.py:749` drew the window's instruction with the global
`random.choice` while the reference pins the draw, and a prompt built from a different instruction is a
different prompt -- 104 tokens against 149 in one measured case. Nothing errors; the loss is simply computed
on another input.

Copying removes that class of problem: the pipeline *is* upstream's, so a difference against upstream is a
difference in upstream, not a difference we introduced. What remains ours is the packing
(`../packing.py`), which cannot be copied: this example trains diffusers' `Cosmos3OmniTransformer`, whose
forward takes loose tensors, while upstream's `OmniMoTModel` takes a `packed_seq` object. The two forward
contracts are different, so the code that produces them has to be.

## Where the copy stops

The line is: **reading the data is upstream's, turning it into model input is ours.**

| stage | code |
| --- | --- |
| DROID index and reading | upstream (`cosmos3_action_lerobot.py`) |
| video decoding | upstream, via torchcodec (`cosmos3_action_lerobot.py:135-138`) |
| instruction pick | upstream (`droid_lerobot_dataset.py:393`) |
| multi-view composition and padding | upstream (`droid_lerobot_dataset.py::_compose_multi_view`) |
| image augmentation | upstream (`droid_lerobot_dataset.py:349`) |
| action, state, idle frames | upstream (`base_dataset.py::_build_result`) |
| prompt assembly | ours (`../packing.py::build_action_json_prompt`) |
| CFG dropout | ours (`../packing.py::tokenize_prompt`) |
| tokenization | ours (`../packing.py`) |
| VAE encode, noising, packing | ours (`../packing.py::pack_one`) |

Upstream has the last four too, in its SFT wrapper (`utils/transforms.py`, `augmentors/text_tokenizer.py`,
tied together by `action_sft_dataset.py`), and they are deliberately not copied. Two reasons.

The first is that the chain cannot be taken without taking `sequence_packing`, upstream's own packing,
which this example cannot use at all: it trains diffusers' `Cosmos3OmniTransformer`, whose forward takes
loose tensors, against upstream's `OmniMoTModel`, whose forward takes a `packed_seq` object. Copying 9949
further lines to reach code that then has to be bypassed buys nothing.

The second is that the three text stages are already known to agree. With the instruction pinned to the
same index on both sides, every tensor the two forwards receive was compared on the same pinned window:
nine of nine identical byte for byte at step 0, including `text_ids` at 170 tokens, and the same nine at
step 1. The prompt this example builds, drops and tokenizes is therefore the prompt upstream builds, drops
and tokenizes -- measured, not assumed, which is the standard the copy itself is held to.

## Checking for drift

Because the files are byte-identical, checking whether upstream has moved is one command against a checkout
of the newer commit:

    diff -r cosmos_framework /path/to/cosmos-framework/cosmos_framework

Empty output means this copy is still exactly upstream's. Any output is either a change to take or a change
to decide about -- it is never ambiguous, which is the point of not editing the files.

To re-copy from a newer commit, list the modules actually reachable from the DROID dataset rather than
guessing. In an environment with upstream installed:

    python -c "import sys, os; \
      import cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset; \
      root = os.getcwd() + '/'; \
      print('\n'.join(sorted(f[len(root):] for n in sys.modules \
        if n.startswith('cosmos_framework') \
        for f in [getattr(sys.modules[n], '__file__', None) or ''] if f.startswith(root))))"

That is how this set of 34 was chosen: it is the transitive closure of the imports, including the sibling
datasets that `datasets/__init__.py` pulls in for other embodiments. Those are inert here -- DROID does not
call them -- but they are kept so that no file had to be edited.

## What was measured before the switch

On one pinned window (episode 26827, first window), with the instruction pinned to index 0 on **both**
sides and augmentation off, this example's own reader was compared against upstream's, field by field.

At the reader's own output:

| field | result |
| --- | --- |
| composite pixels, 3 x 33 x 540 x 640 = 34,214,400 values | **identical**, max abs difference 0.000 levels |
| action rows, and the state row the reader prepends | **identical** |
| instruction | identical |
| idle frames | identical (8) |

Then through the adapter (`../upstream_dataset.py`), which is what `pack_batch` is actually handed, all
thirteen keys with no key missing or extra on either side, and every one identical except `sample_id`:
`video` (so the padding and the [-1, 1] normalisation reproduce exactly), `action`, `state`, `task`,
`idle_frames`, `canvas_hw`, `content_hw`, `aspect_key`, `episode_index`, `start`, `draw_id`,
`additional_view_description`. `sample_id` differs by construction and the reason is in that file.

Two things to be clear about, because "identical" is easy to over-read:

* **The decoders agree here, which does not make them agree everywhere.** PyAV and torchcodec produced the
  same bytes *on the same machine*. The +0.81 LSB recorded elsewhere between this accelerator and x86 is a
  platform difference in the scaler, and this measurement says nothing about it either way.
* **The scaling half of the resize was not exercised, because this dataset cannot reach it.** DROID's three
  640x360 cameras always compose to 540x640, `closest_canvas(540, 640)` is the 4:3 canvas 544x736, and the
  scale factor is capped at 1.0, so the call only ever pads; measured, the returned content region comes
  back as (540, 640), unchanged. The branch is unreachable for DROID rather than untested.
* The augmentation was switched off for the comparison and was not separately verified. It is upstream's
  own code after this change, so what the previous implementation did differently, if anything, now only
  affects runs taken before it.

The entry point then trained two steps on the vendored reader (`--dataset upstream`, one card):
`16338993 window(s)`, `samples=1 AR=168 DM=3093`, loss 24.177132 then 11.412529.

## Why the dependencies are hand-picked, and what that costs

Upstream's pipeline needs `lerobot`, and resolving `lerobot`'s own dependencies is not safe here: the
closure would install `diffusers 0.35.2` over the 0.40.0 that provides `Cosmos3OmniTransformer`, and
`av 15.1.0` over the 18.1.0 whose bundled FFmpeg 8 the decoder links against. Both are load-bearing, so the
packages below were installed individually instead:

| package | version | why |
| --- | --- | --- |
| `lerobot` | 0.4.4 | `LeRobotDataset`, the base of upstream's reader. Installed with `--no-deps`; 0.4.4 is the version upstream builds against |
| `torchcodec` | 0.14.0+cpu | upstream decodes video with it (`cosmos3_action_lerobot.py:135-138` raises if it is absent) |
| `datasets` | 4.8.5 | required by `lerobot`; pinned to the version `lerobot` 0.4.4 was built against rather than the newest |
| `loguru` | latest | required by `cosmos_framework.utils.log` |

`torchcodec` loads FFmpeg by SONAME and finds no system FFmpeg in this image. It ships a variant for
FFmpeg 8 (`libtorchcodec_core8.so`, needing `libavutil.so.60`), which is the version PyAV already bundles
under mangled names, so a directory of correctly-named symlinks onto `av.libs` satisfies it without
changing the image.

**This is the one place the copy does not make us upstream.** Upstream runs inside `lerobot`'s full
dependency set; this runs inside a subset chosen by hand. The code is identical, the environment is not,
and a difference that comes from a package version is still ours to explain. The versions above are pinned
for that reason.
