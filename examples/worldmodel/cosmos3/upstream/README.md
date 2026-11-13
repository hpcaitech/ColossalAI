# The upstream side of the comparison

This example trains Edge on DROID, starting from the base checkpoint `nvidia/Cosmos3-Edge`.
`nvidia/cosmos-framework` publishes an action-policy DROID recipe for Nano only --
`configs/base/experiment/action/posttrain_config/` holds four files and all four are `*_nano.py` -- so there
is no upstream run to compare an Edge step against until one is built. The control group is upstream's
framework, upstream's Nano DROID recipe, Edge's model config and the same base checkpoint.
`action_policy_droid_edge.py` is that overlay: it imports the recipe and the Edge baseline and states three
deltas. Nothing is copied from upstream, so the upstream side of the comparison stays upstream's.

`nvidia/Cosmos3-Edge-Policy-DROID` is not the starting point. It is that base with this very SFT already
applied: compared tensor by tensor, 318 of its 549 transformer tensors differ from the base, and they are
exactly the generation tower (`*_moe_gen`, the `add_*` attention projections, `proj_in`/`proj_out`) plus the
action heads, while the 227 that match are the frozen understanding tower. It is a reference for where
training should arrive, not a place to start.

## What changes because the tier is Edge

Every row below is upstream's own Nano-to-Edge delta, taken from the docstring of `edge_model_config.py`,
except the last one, which is ours.

| Field | Nano | Edge | Effect on the comparison |
| --- | --- | --- | --- |
| `vlm_config.*` (`layer_module`, `model_name`, `pretrained_weights.backbone_path`, `model_instance`) | Qwen3-VL-8B | Nemotron-3 Dense VL, `nvidia/Cosmos3-Edge` | The point of the exercise: 36 blocks of hidden size 4096 become 28 of 2048 |
| `resolution` | `"720"` | `"480"` | None. The rectified-flow shift is looked up by the *sample's* resolution, which the recipe's dataset sets to `"480"` on both sides, so both get shift 5 |
| `rectified_flow_training_config.loss_scale` | `1.0`, raised to `10.0` by the recipe | `10.0` already | None: the same final value. The overlay re-applies the recipe's `10.0` anyway, so the intent is visible rather than inherited |
| `rectified_flow_training_config.image_loss_scale` | `1.0` | `None` | None while the recipe produces no image samples |
| `rectified_flow_training_config.independent_sound_schedule`, `shift_action`, `shift_sound`, `sound_loss_scale` | not set | `False`, `None`, `None`, `2.0` | None. `shift_action=None` inherits the vision shift on both sides, and `sound_gen=False` |
| `max_num_tokens_after_packing` | `45056`, uncapped to `-1` by the recipe | `45056`, uncapped to `-1` by the overlay | None: the same final value |
| `vlm_config.tokenizer` | `nvidia/Cosmos3-Nano` | `nvidia/Cosmos3-Edge`, pinned to the revision the weights come from | **Ours.** This example reads its tokenizer from the checkpoint directory it loads. Two runs tokenizing the prompt with different files would differ in the text segment before the model is reached. Measured: the text tokenizer is byte-identical between `nvidia/Cosmos3-Edge` and `nvidia/Cosmos3-Edge-Policy-DROID`, so this pin keeps the two sides provably on one file rather than fixing a difference we found |
| `checkpoint.keys_to_skip_loading` | `["net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed"]` | unchanged | Left as upstream wrote it: from a base checkpoint the action heads are not DROID-trained, so both sides draw them fresh. `--reinit-action-head` is therefore **on** by default on our side too. Each side draws from its own RNG, so one run exports what it drew (`dump_action_head.py`) and the other loads it (`train_sft.load_action_head`); otherwise the action term of the loss compares two different models |

## What is copied from the Nano recipe unchanged

Optimizer (`FusedAdam`, betas `(0.9, 0.99)`, eps `1e-8`, weight decay `0.05`, lr `2e-4`, the seven
`keys_to_select` prefixes and the three 5x `lr_multipliers`); `LambdaLinear` scheduler; trainer (FSDP,
`grad_accum_iter=1`, `seed=42`, grad clip `1.0`); the dataloader (`PackingDataLoader` with
`max_samples_per_batch=128`, inner `batch_size=16`, `patch_spatial=2`, tokenizer compression 16/4); the
dataset (`fps=15`, `chunk_length=32`, `action_space="joint_pos"`, `mode="wam"`, `use_state=True`,
`iterable_shuffle=True` with `episode_shuffle_seed=42`, `use_image_augmentation=True`,
`viewpoint="concat_view"`, `resolution="480"`, `cfg_dropout_rate=0.1`, `format_prompt_as_json=True`,
`action_normalization=None`, `use_filter_dict=False` at the registered-config level, with the factory's
`use_success_only=True`; the official launch script additionally enables `use_filter_dict=True` with
`keep_ranges_1_0_1.json`); the EMA and tokenizer overrides; `ckpt_type=dcp`; and the recipe's own two model
tweaks, `encode_exact_durations=[33]` and `max_num_tokens_after_packing=-1`.

## What this overlay does not make equal

The optimizer and the parallelism stay different by construction: upstream runs FusedAdam under FSDP, this
example runs HybridAdam under ZeRO-2, and upstream freezes nothing while we freeze everything outside
`TRAINABLE_KEYS`. None of that reaches a forward; all of it is in the way of a backward comparison, and it is
tracked in the differences ledger rather than hidden here.
