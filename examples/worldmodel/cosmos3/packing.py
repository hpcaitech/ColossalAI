"""Turn DROID samples into the arguments of ``Cosmos3OmniTransformer.forward``.

Sequence layout of one sample::

    [ text tokens ][ vision patches ][ action tokens ]
    |- AR, causal -||------- DM, bidirectional ------|

A pack of N samples is laid out as one global AR block followed by one global DM block::

    [ AR_0 | AR_1 | ... | AR_{N-1} | DM_0 | DM_1 | ... | DM_{N-1} ]
      |------- und_len = sum ar_i -||---- per-sample DM blocks ----|

``forward`` carries a single scalar ``und_len``, which is exactly that global split point. The per-sample
boundaries do not fit into any of its arguments, so they travel beside them as a
:class:`~colossalai.shardformer.layer.two_way_attn.TwoWayPackedMeta` that :func:`pack_batch` installs on the
policy itself. See :func:`pack_batch` for why the metadata is installed rather than returned.

Everything here follows ``action_policy_droid_nano`` in nvidia/cosmos-framework. Where a constant or a rule comes
from a specific upstream file, that file is named in the comment, because several of these are the kind of
mismatch that produces a plausible loss curve instead of an error.
"""

from __future__ import annotations

import json
import math

import torch

from colossalai.shardformer.layer import TwoWayPackedMeta

# rectified_flow_training_config, from configs/base/experiment/sft/models/edge_model_config.py
RF_SHIFT = {"256": 3.0, "480": 5.0, "720": 10.0}
WAVER_MODE_S = 1.29  # TrainTimeSampler._WAVER_MODE_S

# Loss weights, same source. The weights are summed as configured and are never renormalized over the modalities
# that happen to be present (algorithm/loss/modality_reduction.py::reduce_global_modality_means).
VISION_LOSS_SCALE = 10.0  # rectified_flow_training_config.loss_scale; the Nano baseline uses 1.0
ACTION_LOSS_WEIGHT = 10.0


def sample_sigma(shift: float, distribution: str = "waver", generator=None) -> float:
    """Sample a training noise level the way ``TrainTimeSampler`` does.

    From model/generator/diffusion/rectified_flow.py::TrainTimeSampler::

        uniform      t = rand()
        logitnormal  t = sigmoid(randn())
        waver        u = rand(); t = 1 - u - S * (cos(pi/2 * u)^2 - 1 + u),  S = 1.29

    followed by the shift ``sigma = shift * t / (1 + (shift - 1) * t)``.

    The Edge configuration uses ``train_time_video_distribution="waver"``,
    ``train_time_action_distribution="logitnormal"``, and ``shift = 5`` at 480p. Note that
    ``rectified_flow_inference_config`` carries a second ``shift = 1``; that one is for inference.

    Args:
        shift (float): Shift applied to the sampled time, from :data:`RF_SHIFT`.
        distribution (str, optional): "uniform", "logitnormal" or "waver". Defaults to "waver".
        generator (torch.Generator, optional): CPU generator, for reproducible runs. Defaults to None.

    Returns:
        float: The noise level sigma in [0, 1).

    Raises:
        ValueError: If the distribution is not one of the three.
    """
    if distribution == "uniform":
        t = torch.rand(1, generator=generator).item()
    elif distribution == "logitnormal":
        t = torch.sigmoid(torch.randn(1, generator=generator)).item()
    elif distribution == "waver":
        u = torch.rand(1, generator=generator).item()
        t = 1.0 - u - WAVER_MODE_S * (math.cos(math.pi / 2.0 * u) ** 2 - 1.0 + u)
    else:
        raise ValueError(f"Unknown distribution {distribution!r}; expected uniform, logitnormal or waver.")
    return shift * t / (1.0 + (shift - 1.0) * t)


def rectified_flow_noise(x0: torch.Tensor, t: torch.Tensor, generator=None):
    """Interpolate towards noise and return the target velocity.

    ``x_t = (1 - sigma) * x0 + sigma * noise`` and the target is ``noise - x0``. The pair is self-consistent in the
    sense that ``x0 = x_t - sigma * target``, which is what tests/test_target_form.py checks: a flipped sign is
    still perfectly learnable on a single sample, so it has to be caught by the identity rather than by training.

    Args:
        x0 (torch.Tensor): Clean sample.
        t (torch.Tensor): Noise level in [0, 1), broadcast against ``x0``.
        generator (torch.Generator, optional): CPU generator. Defaults to None.

    Returns:
        tuple: ``(x_t, target)``, both shaped like ``x0``.
    """
    # The generator lives on CPU, so the noise is drawn there and moved afterwards; this keeps a run reproducible
    # from a seed regardless of the device.
    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype).to(x0.device)
    t = t.to(device=x0.device, dtype=x0.dtype)
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * noise, noise - x0


# ---------------------------------------------------------------------------------------------------------------
# Text prompt: format_prompt_as_json=True and cfg_dropout_rate=0.1
#
# The upstream chain is data/generator/augmentors/text_tokenizer.py::TextTokenizerTransform:
#
#     ActionPromptJsonFormatter(data_dict)  -> a dict
#     json.dumps(dict)                      -> a string
#     with probability 0.1, replace the whole string with ""   <- CFG dropout, not an empty JSON object
#     tokenize_caption(...)                 -> apply_chat_template, user role, add_generation_prompt=True
#     [:4096]                               -> _MAX_NUM_TOKENS
#
# All three steps matter: a plain `tokenizer(task)` produces 9 tokens for a typical DROID task string where this
# chain produces 121, so skipping it changes und_len and every sequence index derived from it.
# ---------------------------------------------------------------------------------------------------------------

CFG_DROPOUT_RATE = 0.1  # dataloader section of action_policy_droid_nano.py
MAX_TEXT_TOKENS = 4096  # text_tokenizer.py::_MAX_NUM_TOKENS

# The action projectors are DomainAwareLinear: one weight/bias pair per embodiment, selected by
# `action_domain_ids`. DROID is 8, from data/generator/action/utils/domain_utils.py::EMBODIMENT_TO_DOMAIN_ID.
# Getting this wrong trains, and writes into, another embodiment's projector while leaving DROID's untouched --
# the loss still falls, so nothing in a training run says anything is wrong.
DROID_DOMAIN_ID = 8

# --- 3D mRoPE position ids -------------------------------------------------------------------------------------
# `position_ids` is [3, sequence_length] interleaved mRoPE, not a flat arange: the three rows are the temporal,
# height and width axes, and Cosmos3VLTextRotaryEmbedding.apply_interleaved_mrope weaves their frequency bands
# together. Giving all three rows the same value degrades it to a 1D rope over sequence position, so the vision
# tokens lose every trace of their 2D layout -- and they were pretrained with one.
#
# Rules, from data/generator/sequence_packing/mrope.py and the call sites in packers.py / sequence.py:
#   * text   -- all three rows share one increasing index, starting at 0 for each sample;
#   * margin -- after the text, the temporal offset jumps by unified_3d_mrope_temporal_modality_margin, so the
#               AR and DM blocks cannot land on overlapping temporal positions;
#   * vision -- a (T, H, W) grid flattened T-major, with the spatial axes restarting at 0 for the segment
#               (unified_3d_mrope_reset_spatial_ids) and the temporal axis scaled to real time;
#   * action -- a (T, 1, 1) grid sharing the vision block's temporal origin, since the two streams run in
#               parallel in time rather than one after the other. Its temporal compression is 1, because actions
#               are at frame rate, but it is normalized against the *vision* base, so the two land on the same
#               time axis. start_frame_offset=1 shifts action i to source frame i+1.
#
# The time scaling is scaled_t = (frame + start_frame_offset) / tps * base_tps, where tps = fps / compression and
# base_tps = MROPE_BASE_FPS / VAE_TEMPORAL_COMPRESSION. At the recipe's 15 fps this puts consecutive latent
# frames 1.6 apart and consecutive actions 0.4 apart, so every 4th action coincides with a latent frame.
MROPE_BASE_FPS = 24.0  # edge_model_config.py::diffusion_expert_config.base_fps
MROPE_TEMPORAL_MARGIN = 15000  # unified_3d_mrope_temporal_modality_margin
VAE_TEMPORAL_COMPRESSION = 4
ACTION_START_FRAME_OFFSET = 1


def build_mrope_position_ids(n_text: int, vision_grid, n_act: int, fps: float) -> torch.Tensor:
    """Build the ``[3, sequence_length]`` mRoPE position ids for one packed sample.

    The layout this matches is ``[text | vision | action]``, the order :func:`pack_one` appends them in.

    Args:
        n_text (int): Number of text tokens.
        vision_grid (tuple): ``(t_latent, hp, wp)``, the patchified latent grid.
        n_act (int): Number of action steps, the conditioning step included.
        fps (float): Frame rate of the source clip, 15 for this recipe.

    Returns:
        torch.Tensor: ``[3, n_text + t*hp*wp + n_act]`` float positions, rows ``(temporal, height, width)``.
    """
    t_latent, hp, wp = vision_grid
    spatial = hp * wp

    # Text: one shared index on all three axes, from 0.
    text = torch.arange(n_text, dtype=torch.float32).unsqueeze(0).expand(3, -1)

    # The generation block starts a margin past the end of the text, never overlapping it.
    offset = float(n_text) + MROPE_TEMPORAL_MARGIN
    base_tps = MROPE_BASE_FPS / VAE_TEMPORAL_COMPRESSION

    # Vision: a T-major (T, H, W) grid. The spatial axes restart at 0 for the segment.
    vision_t = torch.arange(t_latent, dtype=torch.float32) / (fps / VAE_TEMPORAL_COMPRESSION) * base_tps + offset
    vision = torch.stack(
        [
            vision_t.view(-1, 1).expand(-1, spatial).flatten(),
            torch.arange(hp, dtype=torch.float32).view(1, -1, 1).expand(t_latent, -1, wp).flatten(),
            torch.arange(wp, dtype=torch.float32).view(1, 1, -1).expand(t_latent, hp, -1).flatten(),
        ]
    )

    # Action: a (T, 1, 1) grid on the same temporal origin as the vision block, at frame rate (compression 1)
    # but normalized against the vision base, so both streams measure time in the same units.
    action_t = (torch.arange(n_act, dtype=torch.float32) + ACTION_START_FRAME_OFFSET) / fps * base_tps + offset
    action = torch.stack([action_t, torch.zeros_like(action_t), torch.zeros_like(action_t)])

    return torch.cat([text, vision, action], dim=1)


# Verbatim from data/generator/action/utils/viewpoint_utils.py::DEFAULT_VIEWPOINT_TEMPLATES.
VIEWPOINT_TEMPLATES = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": (
        "This video is captured from a third-person perspective looking towards the agent from the front."
    ),
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
    "top_down_2d_view": "This video is captured from a top-down view of a flat 2D scene.",
    "video_game_view": "This video is captured from an in-game camera perspective in a virtual environment.",
}


def _ensure_sentence(text: str) -> str:
    """Add terminal punctuation, matching ``ActionPromptJsonFormatter._ensure_sentence``."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _drop_empty(value):
    """Recursively drop empty strings, lists, dicts and None, matching ``_drop_empty_fields``."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            cleaned = _drop_empty(item)
            if cleaned not in (None, "", [], {}):
                out[key] = cleaned
        return out
    if isinstance(value, list):
        return [c for item in value if (c := _drop_empty(item)) not in (None, "", [], {})]
    return value


def build_action_json_prompt(
    task: str,
    *,
    num_frames: int,
    fps: float,
    canvas_hw,
    aspect: str,
    viewpoint: str = "concat_view",
) -> dict:
    """Build the prompt dictionary with the fields and the order ``ActionPromptJsonFormatter`` emits.

    The field order is fixed upstream and ``json.dumps`` preserves insertion order, so the order reaches the
    tokenizer and must not drift: cinematography, actions, duration, fps, resolution, aspect_ratio.

    Two formatting details that differ from each other on purpose: ``duration`` truncates to whole seconds while
    the end of the ``time`` range rounds, so 33 frames at 15 fps give ``"2s"`` and ``"0:00-0:02"``.

    ``idle_frame`` is omitted. It needs idle-frame counts that come from the ``keep_ranges`` filter table, which the
    recipe leaves off (``use_filter_dict=False``); upstream also drops the field when the count is missing, so the
    result is the same prompt either way.

    Args:
        task (str): The task annotation of this window.
        num_frames (int): Observation frames in the window, used for the duration.
        fps (float): Frame rate.
        canvas_hw: ``(H, W)`` of the canvas **after** padding, not of the content region.
        aspect (str): Aspect key such as ``"4,3"``.
        viewpoint (str, optional): Key into :data:`VIEWPOINT_TEMPLATES`. Defaults to "concat_view".

    Returns:
        dict: The prompt, with empty fields pruned.
    """
    duration = num_frames / float(fps)
    minutes, seconds = divmod(int(round(duration)), 60)
    prompt = {
        "cinematography": {"framing": VIEWPOINT_TEMPLATES.get(viewpoint)},
        "actions": [
            {
                "time": f"0:00-{minutes}:{seconds:02d}",
                "description": _ensure_sentence(task),
            }
        ],
        "duration": f"{int(duration)}s",
        "fps": float(fps),
        "resolution": {"H": int(canvas_hw[0]), "W": int(canvas_hw[1])},
        "aspect_ratio": aspect,
    }
    return _drop_empty(prompt)


def tokenize_prompt(prompt, tokenizer, *, cfg_dropout_rate=CFG_DROPOUT_RATE, generator=None) -> torch.Tensor:
    """Serialize the prompt, optionally drop the condition, and apply the chat template.

    Dropping the condition replaces the **whole string** with an empty one and still runs the chat template, so the
    role markers survive and only the content is gone. It is not a skipped tokenization and not an empty JSON
    object.

    Args:
        prompt: The prompt dictionary from :func:`build_action_json_prompt`, or a plain string.
        tokenizer: A tokenizer with ``apply_chat_template``.
        cfg_dropout_rate (float, optional): Probability of dropping the condition. Defaults to
            :data:`CFG_DROPOUT_RATE`.
        generator (torch.Generator, optional): CPU generator for the dropout draw. Defaults to None.

    Returns:
        torch.Tensor: Token ids, truncated to :data:`MAX_TEXT_TOKENS`.
    """
    text = json.dumps(prompt) if isinstance(prompt, dict) else str(prompt)
    if cfg_dropout_rate > 0 and torch.rand(1, generator=generator).item() < cfg_dropout_rate:
        text = ""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    return torch.tensor(ids[:MAX_TEXT_TOKENS], dtype=torch.long)


def pack_one(
    latent: torch.Tensor,
    action: torch.Tensor,
    input_ids: torch.Tensor,
    cfg,
    *,
    state: torch.Tensor = None,
    fps: float = 15.0,
    domain_id: int = DROID_DOMAIN_ID,
    resolution: str = "480",
    device="npu:0",
    generator=None,
):
    """Pack one sample into the forward arguments. The sound group is omitted, since ``sound_gen`` is off.

    **Conditioning frames.** The recipe runs ``mode="wam"``, and
    data/generator/action/utils/transforms.py::build_sequence_plan_from_mode gives::

        # forward_dynamics/wam/policy: first frame is clean (conditioning)
        condition_frame_indexes_vision = [0]
        # Case B (base_action_length == video_length): history + one initial-state action
        condition_frame_indexes_action = list(range(num_history_actions + 1))

    So latent frame 0 is clean: it is the observation the policy acts on, and noising it leaves the model with
    nothing to condition on. With ``use_state=True`` the action stream is one step longer than the 32 actions,
    the extra leading step encoding the initial robot state, and that step is conditioning too. Passing 32 actions
    and no state is Case A instead, which has no conditioning action at all.

    **What the forward expects of a conditioning frame**, read out of the diffusers implementation. All three have
    to line up; each one fails silently on its own:

    1. ``*_noisy_frame_indexes`` lists only the noisy frames. ``_apply_timestep_embeds_to_noisy_tokens`` scatters
       the timestep embedding onto exactly those, so a conditioning frame receives none and the model reads that
       as "clean".
    2. ``*_timesteps`` is as long as the number of **noisy** tokens, not of all tokens; it is the ``src`` of that
       scatter.
    3. ``*_mse_loss_indexes`` points only at the noisy frames' tokens. ``_unpatchify_and_unpack_latents`` slices
       the MSE head output by ``len(noisy_frame_indexes)``, so a longer index list misaligns it. The returned
       prediction is full size with **zeros at the conditioning frames**, which is why the loss has to multiply by
       the noisy mask instead of ignoring it.

    Args:
        latent (torch.Tensor): ``[1, C, T, H, W]`` VAE latent.
        action (torch.Tensor): ``[chunk, raw_action_dim]``, unnormalized.
        input_ids (torch.Tensor): ``[n_text]`` text tokens.
        cfg: The model config, read for ``latent_patch_size`` and ``action_dim``.
        state (torch.Tensor, optional): ``[raw_action_dim]`` initial robot state, prepended as the conditioning
            action step. Defaults to None, which is Case A.
        fps (float, optional): Frame rate of the source clip, which sets the mRoPE time scale. Defaults to 15.0,
            the recipe value.
        domain_id (int, optional): Selects the per-embodiment action projector weights. Defaults to
            :data:`DROID_DOMAIN_ID`.
        resolution (str, optional): Key into :data:`RF_SHIFT`. Defaults to "480".
        device: Device for the packed tensors.
        generator (torch.Generator, optional): CPU generator. Defaults to None.

    Returns:
        tuple: ``(inputs, targets)``; ``inputs`` goes straight into ``model(**inputs)``.
    """
    to_device = lambda x: x.to(device)  # noqa: E731
    p = cfg.latent_patch_size
    _, _, t_latent, h, w = latent.shape
    hp, wp = h // p, w // p
    spatial = hp * wp
    n_vis = t_latent * spatial

    if state is not None:
        action = torch.cat([state.reshape(1, -1).to(action.dtype), action], dim=0)
    n_act = action.shape[0]
    n_text = input_ids.shape[0]
    seq = n_text + n_vis + n_act

    n_cond_vis = 1
    n_cond_act = 1 if state is not None else 0

    # Pad the action to the model's max_action_dim. Upstream pad_action_to_max_dim pads with zeros, and the loss
    # keeps only the real channels (action_channel_masking), so raw_action_dim has to travel to the loss.
    raw_action_dim = action.shape[-1]
    if raw_action_dim < cfg.action_dim:
        pad = action.new_zeros(n_act, cfg.action_dim - raw_action_dim)
        action = torch.cat([action, pad], dim=-1)

    # Sigma is sampled once and shared by both modalities: independent_action_schedule=False. That is asserted
    # upstream by data/generator/sequence_packing/uses_single_timestep_test.py::
    # test_independent_modality_schedule_invalidates_fast_path, where the action timestep differs from the vision
    # one only when the flag is on. The forward multiplies the timestep by timestep_scale=0.001, so what is passed
    # here is sigma * 1000.
    shift = RF_SHIFT[resolution]
    sigma = sample_sigma(shift, "waver", generator)
    t_vis_raw = t_act_raw = sigma * 1000.0
    latent_noisy, target_vis = rectified_flow_noise(
        latent, torch.tensor(t_vis_raw / 1000.0, dtype=latent.dtype), generator
    )
    action_noisy, target_act = rectified_flow_noise(
        action, torch.tensor(t_act_raw / 1000.0, dtype=action.dtype), generator
    )
    # Restore the conditioning frames. rectified_flow_noise returns fresh tensors, so writing in place here does
    # not touch latent or action.
    latent_noisy[:, :, :n_cond_vis] = latent[:, :, :n_cond_vis]
    action_noisy[:n_cond_act] = action[:n_cond_act]

    vis_idx = to_device(torch.arange(n_text, n_text + n_vis))
    act_idx = to_device(torch.arange(n_text + n_vis, seq))
    n_vis_noisy = (t_latent - n_cond_vis) * spatial
    n_act_noisy = n_act - n_cond_act

    inputs = dict(
        input_ids=to_device(input_ids),
        text_indexes=to_device(torch.arange(n_text)),
        position_ids=to_device(build_mrope_position_ids(n_text, (t_latent, hp, wp), n_act, fps)),
        und_len=n_text,
        sequence_length=seq,
        vision_tokens=[to_device(latent_noisy)],
        vision_token_shapes=[(t_latent, hp, wp)],
        vision_sequence_indexes=vis_idx,
        vision_mse_loss_indexes=vis_idx[n_cond_vis * spatial :],
        vision_timesteps=to_device(torch.full((n_vis_noisy,), t_vis_raw, dtype=latent.dtype)),
        vision_noisy_frame_indexes=[to_device(torch.arange(n_cond_vis, t_latent))],
        action_tokens=[to_device(action_noisy)],
        action_token_shapes=[(n_act, 1, 1)],
        action_sequence_indexes=act_idx,
        action_mse_loss_indexes=act_idx[n_cond_act:],
        action_timesteps=to_device(torch.full((n_act_noisy,), t_act_raw, dtype=action.dtype)),
        action_noisy_frame_indexes=[to_device(torch.arange(n_cond_act, n_act))],
        action_domain_ids=[to_device(torch.tensor(domain_id))],
    )

    # The upstream loss is (sqerr * noisy_mask).mean(): the conditioning rows contribute zero to the numerator
    # while the denominator still counts every element. Slicing the conditioning frames away instead would change
    # the denominator, so the mask travels at full size. See
    # model/generator/algorithm/loss/flow_matching.py::compute_flow_matching_loss.
    mask_vis = target_vis.new_ones(t_latent, 1, 1)
    mask_vis[:n_cond_vis] = 0.0
    mask_act = target_act.new_ones(n_act, 1)
    mask_act[:n_cond_act] = 0.0

    targets = {
        "vision": to_device(target_vis),
        "action": to_device(target_act),
        "raw_action_dim": raw_action_dim,
        "vision_noisy_mask": to_device(mask_vis),
        "action_noisy_mask": to_device(mask_act),
    }
    return inputs, targets


def _as_device(device):
    """``torch.npu.current_device()`` returns an int; normalize it so ``.to()`` accepts it."""
    if isinstance(device, int):
        return torch.device("npu", device)
    return device


def pack_batch(
    batch,
    vae,
    tokenizer,
    cfg,
    *,
    policy,
    num_frames: int,
    fps: float,
    resolution: str = "480",
    device="npu:0",
    generator=None,
):
    """Pack a batch of samples into one sequence and install the per-sample boundaries on the policy.

    Each sample is packed by :func:`pack_one` into ``[text | vision | action]``, i.e. ``[AR_i | DM_i]``, and the
    samples are then rearranged into one global AR block followed by one global DM block. Positions restart at 0
    per sample, so the pack does not encode "which sample this is" into the rotary embedding.

    **Why the metadata is installed here rather than returned.** The processor holds it in a plain attribute and
    can only check the **totals**, not the per-sample split. Two batches colliding on the totals is easy -- the DM
    length is fixed and the AR length is a prompt token count, so (27, 27) and (26, 28) look identical -- and a
    forgotten update would then be accepted silently, attaching one sample's DM tokens to another sample's AR
    prefix without an error.

    That hole cannot be closed inside the processor, because it never sees a forward boundary. The mechanism that
    would provide one, clearing the metadata on the way out, is exactly what rules out a context variable here:
    with ``activation_checkpointing`` in ``"full"`` mode the forward is recomputed during the backward, by which
    time the clearing has already happened. The metadata therefore has to persist, and something that persists
    cannot tell stale from fresh. The only place left to close it is the caller, by making packing and installing
    one call that cannot be half-performed.

    Args:
        batch: A collated batch, either with ``video`` or with a precomputed ``latent``.
        vae: The VAE, or None when the batch carries latents.
        tokenizer: Tokenizer for the prompt.
        cfg: The model config.
        policy: The Cosmos3 policy, or anything exposing ``set_packed_meta``.
        num_frames (int): Observation frames per window, for the prompt duration.
        fps (float): Frame rate, for the prompt duration.
        resolution (str, optional): Key into :data:`RF_SHIFT`. Defaults to "480".
        device: Device for the packed tensors.
        generator (torch.Generator, optional): CPU generator. Defaults to None.

    Returns:
        tuple: ``(inputs, targets)``.

    Raises:
        ValueError: If the batch has neither latents nor a usable VAE.
    """
    device = _as_device(device)
    n = batch["latent"].shape[0] if "latent" in batch else batch["video"].shape[0]

    if "latent" in batch:
        latent = batch["latent"].to(device=device, dtype=torch.bfloat16)
    else:
        if vae is None:
            raise ValueError("The batch carries no latent and no VAE was provided; supply one or the other.")
        # The VAE can only run in the main process, since workers cannot touch the device. Encode the batch in one
        # call and slice it per sample afterwards.
        with torch.no_grad():
            out = vae.encode(batch["video"].to(device, torch.float32))
            dist = getattr(out, "latent_dist", None)
            if dist is not None:
                latent = dist.mode()
            elif isinstance(out, (tuple, list)):
                latent = out[0]
            else:
                latent = out
            latent = latent.to(torch.bfloat16)

    per_sample = []
    for index in range(n):
        prompt = build_action_json_prompt(
            batch["task"][index],
            num_frames=num_frames,
            fps=fps,
            canvas_hw=batch["canvas_hw"][index].tolist(),
            aspect=batch["aspect_key"][index],
        )
        ids = tokenize_prompt(prompt, tokenizer, generator=generator)
        per_sample.append(
            pack_one(
                latent[index : index + 1],
                batch["action"][index].to(torch.bfloat16),
                ids,
                cfg,
                state=batch["state"][index].to(torch.bfloat16) if "state" in batch else None,
                fps=fps,
                device=device,
                resolution=str(resolution),
                generator=generator,
            )
        )

    ar_lens = [int(inputs["und_len"]) for inputs, _ in per_sample]
    dm_lens = [int(inputs["sequence_length"]) - int(inputs["und_len"]) for inputs, _ in per_sample]
    total_ar, total_dm = sum(ar_lens), sum(dm_lens)
    sequence_length = total_ar + total_dm

    # Take the dtype from what pack_one produced rather than naming one here. The mRoPE temporal axis is
    # fractional under fps modulation, and a torch.long buffer would truncate it on assignment -- silently,
    # since the shapes still line up.
    position_ids = torch.zeros(3, sequence_length, dtype=per_sample[0][0]["position_ids"].dtype, device=device)
    input_ids, text_indexes = [], []
    vision_indexes, vision_loss_indexes = [], []
    action_indexes, action_loss_indexes = [], []
    vision_timesteps, action_timesteps = [], []
    per_item_keys = (
        "vision_tokens",
        "vision_token_shapes",
        "vision_noisy_frame_indexes",
        "action_tokens",
        "action_token_shapes",
        "action_noisy_frame_indexes",
        "action_domain_ids",
    )
    merged = {key: [] for key in per_item_keys}
    targets = {
        "vision": [],
        "action": [],
        "raw_action_dim": None,
        "vision_noisy_mask": [],
        "action_noisy_mask": [],
    }

    ar_offset, dm_offset = 0, total_ar
    for index, (inputs, sample_targets) in enumerate(per_sample):
        n_text, n_dm = ar_lens[index], dm_lens[index]

        text_indexes.append(torch.arange(ar_offset, ar_offset + n_text, device=device))
        position_ids[:, ar_offset : ar_offset + n_text] = inputs["position_ids"][:, :n_text]
        position_ids[:, dm_offset : dm_offset + n_dm] = inputs["position_ids"][:, n_text:]

        # pack_one numbers its DM indexes from its own origin at n_text; rebase them onto the global DM block.
        shift = dm_offset - n_text
        vision_indexes.append(inputs["vision_sequence_indexes"] + shift)
        vision_loss_indexes.append(inputs["vision_mse_loss_indexes"] + shift)
        action_indexes.append(inputs["action_sequence_indexes"] + shift)
        action_loss_indexes.append(inputs["action_mse_loss_indexes"] + shift)

        input_ids.append(inputs["input_ids"])
        vision_timesteps.append(inputs["vision_timesteps"])
        action_timesteps.append(inputs["action_timesteps"])
        for key in per_item_keys:
            merged[key].extend(inputs[key])

        targets["vision"].append(sample_targets["vision"])
        targets["action"].append(sample_targets["action"])
        targets["vision_noisy_mask"].append(sample_targets["vision_noisy_mask"])
        targets["action_noisy_mask"].append(sample_targets["action_noisy_mask"])
        targets["raw_action_dim"] = sample_targets["raw_action_dim"]

        ar_offset += n_text
        dm_offset += n_dm

    packed = dict(
        input_ids=torch.cat(input_ids),
        text_indexes=torch.cat(text_indexes),
        position_ids=position_ids,
        und_len=total_ar,
        sequence_length=sequence_length,
        vision_sequence_indexes=torch.cat(vision_indexes),
        vision_mse_loss_indexes=torch.cat(vision_loss_indexes),
        vision_timesteps=torch.cat(vision_timesteps),
        action_sequence_indexes=torch.cat(action_indexes),
        action_mse_loss_indexes=torch.cat(action_loss_indexes),
        action_timesteps=torch.cat(action_timesteps),
        **merged,
    )
    policy.set_packed_meta(TwoWayPackedMeta(ar_lens, dm_lens))
    return packed, targets


def compute_loss(out, targets):
    """Flow-matching loss over the vision and action modalities.

    Written against model/generator/algorithm/loss/flow_matching.py::compute_flow_matching_loss::

        per_sample = (sqerr * noisy_mask).mean()   # the denominator counts every element, not only noisy ones
        modality   = stack(per_sample).mean()      # averaged per sample, not per token
        total      = sum(weight[m] * modality_m)   # fixed weights, never renormalized

    The mask is load bearing: the prediction comes back at full size with zeros at the conditioning frames, so
    without it those rows enter the loss as ``(0 - target)**2``. Measured on one real batch, dropping the mask
    inflates the vision term by about 1.2x. The denominator has to stay at full size, so slicing the conditioning
    frames away is not equivalent either; upstream is ``.mean()``, not ``sum / active``.

    The action term keeps only the first ``raw_action_dim`` channels, since pack_one pads the action out to the
    model's ``action_dim`` and the padded channels carry no supervision (``action_channel_masking``).

    There is no per-timestep weighting: ``train_time_weight="uniform"`` resolves to ``torch.ones_like(t)`` in
    model/generator/algorithm/loss/time_weight.py, which is the only weight the rectified-flow path implements.

    Args:
        out: The transformer output, with ``sample`` and ``action`` lists.
        targets (dict): The targets from :func:`pack_batch`.

    Returns:
        torch.Tensor: The scalar loss.

    Raises:
        ValueError: If the model returned no vision or action predictions.
    """
    if not out.sample or not out.action:
        raise ValueError(
            "The model returned no vision or action predictions; check that the forward received non-empty "
            "vision and action inputs."
        )

    vision = torch.stack(
        [
            ((pred.float() - target.float()).pow(2) * mask.float()).mean()
            for pred, target, mask in zip(out.sample, targets["vision"], targets["vision_noisy_mask"])
        ]
    ).mean()
    dim = targets["raw_action_dim"]
    action = torch.stack(
        [
            ((pred[:, :dim].float() - target[:, :dim].float()).pow(2) * mask.float()).mean()
            for pred, target, mask in zip(out.action, targets["action"], targets["action_noisy_mask"])
        ]
    ).mean()
    return vision * VISION_LOSS_SCALE + action * ACTION_LOSS_WEIGHT
