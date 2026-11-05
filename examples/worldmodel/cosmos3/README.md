# Cosmos 3 Edge action-policy SFT on Ascend

Supervised fine-tuning of **Cosmos3-Edge-Policy-DROID** with ColossalAI, following the
`action_policy_droid_nano` recipe of [nvidia/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) with
Edge's model config.

Cosmos 3 is a mixture-of-transformers world model: a Qwen-style text tower (the autoregressive, *understanding*
pathway) and a diffusion tower (the *generation* pathway) share one stack of decoder layers and one packed
sequence. The AR tower stays frozen here; the generation tower and the action heads train, which is 1.42B of the
model's 3.37B parameters.

The attention over that shared sequence is asymmetric — AR is causal, DM is bidirectional over itself and sees
the AR prefix, AR never sees DM — and is provided by `colossalai.shardformer.policies.cosmos3.Cosmos3OmniPolicy`,
which installs a processor backed by the two-way flat-attention kernel.

## What you need

- An Ascend 910B host with `torch_npu`, or a CUDA host (the code reads the communication backend off the
  accelerator rather than hard-coding one, but it has only been run on Ascend).
- The `Cosmos3-Edge-Policy-DROID` checkpoint, with its `transformer/`, `vae/` and `text_tokenizer/`
  subdirectories.
- DROID in LeRobot v3 layout (`meta/`, `data/`, `videos/`), for the full data path.

```bash
pip install -r requirements.txt
```

`diffusers` must be new enough to ship `Cosmos3OmniTransformer`; 0.40.0 does.

## Running

```bash
python -m torch.distributed.run --nproc_per_node=8 train_sft.py \
    --data /path/to/Cosmos3-DROID/success \
    --checkpoint /path/to/Cosmos3-Edge-Policy-DROID \
    --out runs/cosmos3-sft
```

The defaults are the recipe values: 8 ranks x 32 samples x 32 accumulation steps is the global batch of 8192 the
256-rank upstream run uses, at `lr=2e-4` with the action heads at 5x that.

Use `python -m torch.distributed.run` rather than `torchrun`. In a virtualenv that inherits torch from the system
interpreter, `torchrun` resolves to that interpreter's script and ColossalAI will not be importable there.

A short single-rank run:

```bash
python -m torch.distributed.run --nproc_per_node=1 train_sft.py \
    --data ... --checkpoint ... \
    --samples-per-rank 1 --grad-accum 1 --max-iter 60 --max-episodes 4
```

To exercise the loop with no video and no VAE, pass `--latents` a safetensors file whose entries are
`<name>/latent` and `<name>/action` and whose header records `canvas_hw` and `aspect_key` per clip.

## The parts that are easy to get wrong

Each of these fails silently — a plausible loss curve rather than an error — so each one is named here with the
upstream file it comes from.

**The first latent frame is a clean conditioning frame.** The recipe runs `mode="wam"`, and
`build_sequence_plan_from_mode` gives `condition_frame_indexes_vision = [0]`. It is the observation the policy
acts on; noising it leaves the model nothing to condition on. With `use_state=True` the action stream carries one
extra leading step holding the initial robot state, and that step is conditioning too.

The forward has three separate expectations about a conditioning frame, and all three have to agree:
`*_noisy_frame_indexes` lists only noisy frames (the timestep embedding is scattered onto exactly those, and a
frame without one is read as clean); `*_timesteps` is sized by noisy tokens rather than all tokens; and
`*_mse_loss_indexes` points only at noisy frames, because the MSE head output is sliced by
`len(noisy_frame_indexes)`. The prediction comes back at full size with **zeros at the conditioning frames**, so
the loss multiplies by a noisy mask while keeping the full denominator — dropping the mask inflates the vision
term by about 1.2x.

**The state must be in the action space.** `action_space="joint_pos"` means joint(7) + gripper(1), so the
conditioning step reads `observation.state.joint_positions` and `observation.state.gripper_position`, not
`cartesian_position`.

**The action domain id is 8.** `action_proj_in` and `action_proj_out` are `DomainAwareLinear`: an `nn.Embedding`
of weight/bias pairs indexed by `action_domain_ids`, one per embodiment. DROID's slot is 8, from
`domain_utils.py::EMBODIMENT_TO_DOMAIN_ID["droid_lerobot"]`. A wrong id trains another embodiment's projector and
leaves DROID's at its pretrained value, and the loss curve is indistinguishable, because the projector it picked
up is trainable too. Note that `EMBODIMENT_TO_RAW_ACTION_DIM["droid_lerobot"]` in the same file is 10, for the
`ee_pose` space of `[Pos, Rot6d, Gripper]`; `joint_pos` is 8 wide. Two different 8s and a 10 in one table.

**The noise schedule.** `train_time_video_distribution="waver"` (not the library default `uniform`), `shift=5`
at 480p (`rectified_flow_inference_config` carries a second `shift=1`, which is for inference), and
`independent_action_schedule=False`, so the action and the vision share one sigma.

**The text prompt is JSON through a chat template.** `ActionPromptJsonFormatter` builds a dict with a fixed field
order, `json.dumps` turns it into a string, 10% of samples have that string replaced with `""`
(`cfg_dropout_rate`), and the result goes through `apply_chat_template` with the user role and a generation
prompt. A plain `tokenizer(task)` gives 9 tokens where this chain gives 121, which changes `und_len` and every
index derived from it.

**HF32 has to be off for the VAE.** With `torch.npu.conv.allow_hf32` left at its default, an fp32 convolution was
measured at 210x the CPU error, which quietly corrupts the latents.

## Layout

| Path | What it is |
| --- | --- |
| `droid_dataset.py` | Map-style DROID dataset: `concat_view` composition, canvas fitting, reflection padding |
| `packing.py` | Noise schedule, JSON prompt, packing into the forward arguments, the loss |
| `train_sft.py` | Training entry point, the precomputed-latent bypass, the ZeRO-2 plugin wiring |
| `tests/` | CPU-only checks on the target form, the loss masking and the prompt |
| `scripts/` | Smoke scripts that need a device, the checkpoint and the dataset |

## Tests

`tests/` runs anywhere:

```bash
pytest tests/
COSMOS3_CHECKPOINT=/path/to/Cosmos3-Edge-Policy-DROID pytest tests/   # also runs the tokenizer checks
```

`tests/test_target_form.py` checks what training cannot: that `x0 = x_t - sigma * target` holds on the noisy
frames (a flipped sign breaks it), that the conditioning frame is excluded by the mask rather than by zeroing,
that a perfect prediction gives exactly zero loss, and that the action domain id is DROID's.

`scripts/` needs the real thing:

```bash
python scripts/smoke_conditioning.py --data ... --checkpoint ...
python -m torch.distributed.run --nproc_per_node=1 scripts/smoke_overfit.py --data ... --checkpoint ...
```

`smoke_overfit.py` fixes sigma, packs one sample once and feeds it repeatedly, so the problem degenerates into a
deterministic regression and the loss must fall by orders of magnitude. It shows the optimization path is
connected; it does not show the target is correct, because a single sample is learnable under a wrong target too.

## Status

Measured on one 910B2, single rank:

| | |
| --- | --- |
| Single-sample training step through `booster` + ZeRO-2 | works, ~2.3 s/step at the recipe window |
| Overfitting one sample, 200 steps | 30.34 -> 0.0031 |
| Fused kernel vs the PyTorch reference, in the training loop | relative difference 2e-5 |

Not done yet: there is no comparison against cosmos-framework on a reference device, so the recipe is verified to
be *self-consistent* and to match what the upstream source says, not to produce the same model.

`--backend naive` is the PyTorch reference. It materializes the full score matrix and runs out of memory at the
recipe's sequence length, so it is for numerical comparison at a shortened window, not for training.

The image augmentation is a random resized crop only; upstream also applies a ColorJitter shared across the three
views.
