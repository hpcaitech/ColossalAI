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

import contextlib
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

    Everything here is fp32, whatever dtype ``x0`` arrives in, because that is how upstream does it: it keeps
    the clean payload in fp32 inside its ``PackedSequence`` and noises after packing, so its noise, its ``x_t``
    and its target are all fp32. Drawing the noise in bf16 instead -- which is what ``dtype=x0.dtype`` used to
    do, the payload coming out of a bf16 VAE -- quantizes the samples at about 0.4% relative, and the target
    ``noise - x0`` inherits that. The target is what the gradient is taken against, so it is the one tensor
    here worth keeping exact; the caller casts ``x_t`` to the model's dtype on its way into the forward and
    leaves the target alone.

    Args:
        x0 (torch.Tensor): Clean sample, in any floating dtype.
        t (torch.Tensor): Noise level in [0, 1), broadcast against ``x0``.
        generator (torch.Generator, optional): CPU generator. Defaults to None.

    Returns:
        tuple: ``(x_t, target)``, both shaped like ``x0`` and both fp32.
    """
    # The generator lives on CPU, so the noise is drawn there and moved afterwards; this keeps a run reproducible
    # from a seed regardless of the device.
    noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device)
    x0 = x0.float()
    t = t.to(device=x0.device, dtype=torch.float32)
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


def action_start_frame_offset(*, has_initial_state: bool) -> int:
    """Which vision frame ``action[0]`` is aligned to, in the mRoPE temporal axis.

    Upstream derives this rather than fixing it (``transforms.py::build_sequence_plan``), and the derivation is a
    case split on whether the action stream is as long as the video. With ``num_history_actions = 0``, which is
    this recipe:

    ===========================================  ======  =============================
    condition                                    offset  meaning
    ===========================================  ======  =============================
    ``base_action_length == video_length - 1``        1  no initial-state row
    ``base_action_length == video_length``            0  initial-state row present
    ===========================================  ======  =============================

    The ``+1`` exists only to skip past the action slot that an initial state would otherwise occupy
    (``sequence.py`` documents the field as "align action[0] with vision frame action_start_frame_offset"). With
    ``use_state=True`` the state *is* action row 0 and belongs at frame 0, next to the conditioning vision frame.

    This module used to hardcode 1 while prepending the state row, which placed every action token one frame
    (1/15 s) late. The upstream sample for episode 26827 carries ``sequence_plan.action_start_frame_offset = 0``,
    which is where the value here comes from -- read off a real sample rather than off the default in a signature.

    Args:
        has_initial_state (bool): Whether the initial state has been prepended as action row 0.

    Returns:
        int: 0 with the state row, 1 without it.
    """
    return 0 if has_initial_state else 1


def build_mrope_position_ids(
    n_text: int, vision_grid, n_act: int, fps: float, *, action_frame_offset: int
) -> torch.Tensor:
    """Build the ``[3, sequence_length]`` mRoPE position ids for one packed sample.

    The layout this matches is ``[text | vision | action]``, the order :func:`pack_one` appends them in.

    Args:
        n_text (int): Number of text tokens.
        vision_grid (tuple): ``(t_latent, hp, wp)``, the patchified latent grid.
        n_act (int): Number of action steps, the conditioning step included.
        fps (float): Frame rate of the source clip, 15 for this recipe.
        action_frame_offset (int): The vision frame ``action[0]`` aligns to, from
            :func:`action_start_frame_offset`. Deliberately has no default: the whole point of the helper is that
            the value depends on the sample's layout, and a default here is what hid a one-frame shift before.

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
    action_t = (torch.arange(n_act, dtype=torch.float32) + action_frame_offset) / fps * base_tps + offset
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
    additional_view_description: str | None = None,
    idle_frames: int | None = None,
    total_action_frames: int | None = None,
) -> dict:
    """Build the prompt dictionary with the fields and the order ``ActionPromptJsonFormatter`` emits.

    The field order is fixed upstream and ``json.dumps`` preserves insertion order, so the order reaches the
    tokenizer and must not drift: cinematography, actions, duration, fps, resolution, aspect_ratio.

    Two formatting details that differ from each other on purpose: ``duration`` truncates to whole seconds while
    the end of the ``time`` range rounds, so 33 frames at 15 fps give ``"2s"`` and ``"0:00-0:02"``.

    ``framing`` is two layers. :data:`VIEWPOINT_TEMPLATES` is the generic table every dataset shares, and each
    dataset appends its own sentences through ``extras["additional_view_description"]``, which
    ``ActionPromptJsonFormatter._get_viewpoint_caption`` joins on. The separator rule is upstream's: a single space
    when the template already ends in a period, ``". "`` otherwise.

    ``idle_frame`` is emitted whenever a count is supplied. An earlier revision of this docstring claimed the count
    came from the ``keep_ranges`` filter table and was therefore unavailable with ``use_filter_dict=False``; both
    halves were wrong. The count is computed from the raw action chunk by
    ``pose_utils.compute_idle_frames`` (:func:`droid_dataset.count_idle_frames` here), the field is gated only on
    ``mode != "inverse_dynamics"``, and upstream emits it for this recipe. ``keep_ranges`` drops whole windows from
    the index; ``idle_frames`` counts still frames inside one window. The two share the word "idle" and nothing else.

    Args:
        task (str): The task annotation of this window.
        num_frames (int): Observation frames in the window, used for the duration.
        fps (float): Frame rate.
        canvas_hw: ``(H, W)`` of the canvas **after** padding, not of the content region.
        aspect (str): Aspect key such as ``"4,3"``.
        viewpoint (str, optional): Key into :data:`VIEWPOINT_TEMPLATES`. Defaults to "concat_view".
        additional_view_description (str, optional): The dataset's own view sentences, appended to the generic
            template. Defaults to None.
        idle_frames (int, optional): Idle frames in this window's action chunk. Omits the field when None.
            Defaults to None.
        total_action_frames (int, optional): Action rows the count is out of, for the ``"N out of M."`` form.
            Upstream falls back to ``"N."`` without it. Defaults to None.

    Returns:
        dict: The prompt, with empty fields pruned.
    """
    duration = num_frames / float(fps)
    minutes, seconds = divmod(int(round(duration)), 60)
    framing = VIEWPOINT_TEMPLATES.get(viewpoint)
    if additional_view_description:
        if framing is None:
            framing = str(additional_view_description).rstrip()
        else:
            framing = framing + (" " if framing.endswith(".") else ". ") + str(additional_view_description).rstrip()
    action_entry = {
        "time": f"0:00-{minutes}:{seconds:02d}",
        "description": _ensure_sentence(task),
    }
    if idle_frames is not None:
        # Key order matters: upstream appends idle_frame after description, and json.dumps preserves insertion
        # order all the way to the tokenizer.
        action_entry["idle_frame"] = (
            f"{int(idle_frames)} out of {int(total_action_frames)}."
            if total_action_frames is not None
            else f"{int(idle_frames)}."
        )
    prompt = {
        "cinematography": {"framing": framing},
        "actions": [action_entry],
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
    return torch.tensor(close_text_span(ids[:MAX_TEXT_TOKENS], tokenizer), dtype=torch.long)


# The token that tells the model generation follows. Upstream calls it start_of_generation and takes it from
# the tokenizer by name (sequence_packing/modality.py::add_special_tokens).
START_OF_GENERATION = "<|vision_start|>"


def close_text_span(ids: list[int], tokenizer) -> list[int]:
    """Append the two markers upstream's text split ends with: EOS, then start-of-generation.

    ``PackedSequenceBuilder.pack_text_tokens`` builds the span as

        [BOS if the special-token dict has one] + caption_ids + [EOS] + [start_of_generation if media follows]

    and this recipe always has media following. Upstream's dict comes from ``add_special_tokens``, which
    defines ``start_of_generation`` / ``end_of_generation`` and no ``bos_token_id``, so no BOS is prepended;
    measured, our 149 caption tokens are upstream's first 149 and the two it adds are these.

    Leaving them off is not cosmetic. The BOV token is the boundary the model reads between the prompt and the
    generated block, and without the pair every index after the text -- vision, action, loss, mRoPE positions --
    sits two places early relative to upstream's.
    """
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("the tokenizer defines no eos_token_id; upstream's text span always ends with EOS")
    bov = tokenizer.convert_tokens_to_ids(START_OF_GENERATION)
    if bov is None or bov == getattr(tokenizer, "unk_token_id", object()):
        raise ValueError(f"the tokenizer does not know {START_OF_GENERATION}, which marks the generated block")
    return list(ids) + [int(eos), int(bov)]


def pack_one(
    latent: torch.Tensor,
    action: torch.Tensor,
    input_ids: torch.Tensor,
    cfg,
    *,
    state: torch.Tensor = None,
    payload_dtype: torch.dtype = torch.bfloat16,
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
    # Ceiling, not floor: an odd latent side is zero-padded up to the patch grid, never truncated. The forward
    # does exactly that -- ``_patchify_and_pack_latents`` computes ``h_padded = ceil(h / p) * p``, zero-fills the
    # new row, and derives its own ``h_patches`` from it -- and upstream's packed sequence reserves the same
    # ceil-sized grid (token_shapes (9, 17, 20) for a 33x40 latent). Flooring would reserve 16 rows of sequence
    # for the 17 the forward produces, and it only became reachable once the latent stopped being padded to the
    # canvas: 544/16 = 34 is even, the 540-row content gives 33.
    hp, wp = -(-h // p), -(-w // p)
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
    # The dtype x_t leaves in is the model's, not the payload's. Keeping the payload in fp32 all the way to
    # here is what the reference implementation does -- measured on its own run, ``x0_tokens_vision`` and
    # ``x0_tokens_action`` are float32 even when the model is bf16, and only ``xt_tokens_*`` is bf16 -- and it
    # matters: rounding x0 before the interpolation throws away precision the interpolation would have kept,
    # and no fp32 arithmetic downstream can recover it. Measured from one shared x0, with the payload left in
    # fp32 our x_t and both targets come out bit-identical to upstream's.
    latent_dtype = action_dtype = payload_dtype
    shift = RF_SHIFT[resolution]
    sigma = sample_sigma(shift, "waver", generator)
    t_vis_raw = t_act_raw = sigma * 1000.0
    # fp32 for sigma too, not the payload dtype: bf16 cannot hold a sampled sigma. The pinned 0.5 used in the
    # upstream comparison is exact in bf16, which is why a comparison at that sigma would not have shown it.
    latent_noisy, target_vis = rectified_flow_noise(
        latent, torch.tensor(t_vis_raw / 1000.0, dtype=torch.float32), generator
    )
    action_noisy, target_act = rectified_flow_noise(
        action, torch.tensor(t_act_raw / 1000.0, dtype=torch.float32), generator
    )
    # Restore the conditioning frames. rectified_flow_noise returns fresh tensors, so writing in place here does
    # not touch latent or action. Those tensors are fp32 now and the clean rows come from the payload's own
    # dtype, so the assignment widens them, which is exact.
    latent_noisy[:, :, :n_cond_vis] = latent[:, :, :n_cond_vis]
    action_noisy[:n_cond_act] = action[:n_cond_act]
    # And clear the padding channels the noise just filled. Upstream noises the full padded width and then
    # throws that part away (omni_mot_model.py::_add_noise_to_input):
    #
    #     for i in range(len(xt_action)):
    #         if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
    #             xt_action[i][:, gen_data_clean.raw_action_dim[i] :] = 0
    #
    # Those channels are not a formality. ``action_proj_in`` is a 64 -> 4096 projection, so noise sitting in
    # the 56 padded channels is projected into the hidden state like any other input: measured on episode
    # 26827 offset 0 at a pinned sigma of 0.5, our action rows entering block 0 carried 52% more norm than
    # upstream's (165.8 against 108.8, relL2 1.15), while the conditioning row -- which is never noised --
    # matched bit for bit, and the vision rows, which have no padding, agreed to 3.3%. With the padding
    # cleared, our x_t matches upstream's to relL2 1.6e-03, the bf16 floor.
    #
    # Only x_t is cleared, which is what upstream clears. The target keeps its padding, and it never reaches
    # the loss: compute_loss crops both sides to raw_action_dim.
    action_noisy[:, raw_action_dim:] = 0

    vis_idx = to_device(torch.arange(n_text, n_text + n_vis))
    act_idx = to_device(torch.arange(n_text + n_vis, seq))
    n_vis_noisy = (t_latent - n_cond_vis) * spatial
    n_act_noisy = n_act - n_cond_act

    inputs = dict(
        input_ids=to_device(input_ids),
        text_indexes=to_device(torch.arange(n_text)),
        position_ids=to_device(
            build_mrope_position_ids(
                n_text,
                (t_latent, hp, wp),
                n_act,
                fps,
                # n_cond_act is 1 exactly when the state row was prepended, which is upstream's
                # has_initial_state.
                action_frame_offset=action_start_frame_offset(has_initial_state=bool(n_cond_act)),
            )
        ),
        und_len=n_text,
        sequence_length=seq,
        # x_t goes in at the model's dtype -- the payload's, which is the VAE's -- while the target above stays
        # fp32. Upstream does the same: it noises in fp32 and the forward casts on the way in.
        vision_tokens=[to_device(latent_noisy.to(latent_dtype))],
        vision_token_shapes=[(t_latent, hp, wp)],
        vision_sequence_indexes=vis_idx,
        vision_mse_loss_indexes=vis_idx[n_cond_vis * spatial :],
        # fp32, not the payload dtype. The forward computes ``time_proj(timesteps * timestep_scale)`` and only
        # casts afterwards, and upstream carries these in fp32; in bf16 a sampled timestep of 734.21 is
        # stored as 734.0, which shifts the embedding of every noisy token. The pinned sigma 0.5 -> 500.0 is
        # exact in bf16, so a comparison at that sigma would not have shown it.
        vision_timesteps=to_device(torch.full((n_vis_noisy,), t_vis_raw, dtype=torch.float32)),
        vision_noisy_frame_indexes=[to_device(torch.arange(n_cond_vis, t_latent))],
        action_tokens=[to_device(action_noisy.to(action_dtype))],
        action_token_shapes=[(n_act, 1, 1)],
        action_sequence_indexes=act_idx,
        action_mse_loss_indexes=act_idx[n_cond_act:],
        action_timesteps=to_device(torch.full((n_act_noisy,), t_act_raw, dtype=torch.float32)),
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


def normalize_latent(latent: torch.Tensor, vae) -> torch.Tensor:
    """Scale a raw VAE latent by the Wan 2.2 channel statistics, the way the upstream encode does.

    Upstream never hands out a raw latent: ``WanVAE.__init__`` builds ``self.scale = (mean, 1.0 / std)`` from 48
    hardcoded Wan 2.2 channel statistics, and ``Wan2pt2VAEInterface.encode`` calls
    ``self.model.encode(videos, self.scale)``, so the tensor the transformer sees is already whitened per
    channel. diffusers keeps the same statistics on the config as ``latents_mean`` / ``latents_std`` but does not
    apply them inside ``encode`` -- that is left to the caller, and we were not doing it.

    Measured on one real DROID window against the upstream encode, same pixels, same GPU:

        raw latent                    relL2 5.1652e-01   std 0.683105
        (x - mean) * (1 / std)        relL2 5.8847e-03   std 1.017034
        upstream                                  --     std 1.017380

    The 48 channel constants are identical on both sides: diffusers' ``latents_mean`` / ``latents_std`` match
    upstream's hardcoded lists element for element, all 48 of each. So the only thing this function has to get
    right is the formula, and the residual 5.9e-03 is the bf16 encode rather than the scaling.

    So the model was being fed latents off by a per-channel affine map, a relative error of 0.52 -- 84% of the
    augmentation spread, and two orders of magnitude worse than anything else measured in this example. The
    residual 5.9e-03 after scaling is the bf16 encode itself (4.8e-03 on 910B, 6.4e-03 on H20), so this closes
    the gap down to the precision floor.

    The statistics are read off the VAE config rather than hardcoded, so they stay tied to the checkpoint.

    Args:
        latent (torch.Tensor): ``[B, C, T, H, W]`` raw latent straight out of ``vae.encode``.
        vae: The VAE, read for ``config.latents_mean`` and ``config.latents_std``.

    Returns:
        torch.Tensor: The normalized latent, in the dtype of ``latent``.
    """
    cfg = getattr(vae, "config", None)
    mean = getattr(cfg, "latents_mean", None)
    std = getattr(cfg, "latents_std", None)
    if mean is None or std is None:
        raise ValueError(
            "The VAE config carries no latents_mean/latents_std, so the latent cannot be scaled the way "
            "upstream scales it. Encoding without them leaves the latent off by a per-channel affine map."
        )
    shape = (1, -1) + (1,) * (latent.ndim - 2)
    m = torch.as_tensor(mean, device=latent.device, dtype=torch.float32).view(shape)
    # Reciprocal first, then multiply, because that is literally what upstream does: ``WanVAE.__init__`` builds
    # ``self.scale = (mean, 1.0 / std)`` and multiplies by it ever after. The choice is not numerically load
    # bearing -- measured on a latent of the real shape, ``(x - m) / s`` and ``(x - m) * (1 / s)`` differ by
    # relL2 4.17e-08, max 4.8e-07, on 24% of elements, which is one float32 ulp. It is written this way so the
    # correspondence to upstream's formula needs no argument, not because it moves any number.
    inv_std = 1.0 / torch.as_tensor(std, device=latent.device, dtype=torch.float32).view(shape)
    return ((latent.float() - m) * inv_std).to(latent.dtype)


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
    payload_dtype: torch.dtype = torch.bfloat16,
    stopwatch=None,
    vae_encode_batch: int = 0,
    latents=None,
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
        batch: A collated batch carrying ``video``.
        vae: The VAE. Required: the latent is encoded here.
        tokenizer: Tokenizer for the prompt.
        cfg: The model config.
        policy: The Cosmos3 policy, or anything exposing ``set_packed_meta``.
        num_frames (int): Observation frames per window, for the prompt duration.
        fps (float): Frame rate, for the prompt duration.
        resolution (str, optional): Key into :data:`RF_SHIFT`. Defaults to "480".
        device: Device for the packed tensors.
        generator (torch.Generator, optional): CPU generator. Defaults to None.
        stopwatch: Optional section timer; records the online encode separately.
        vae_encode_batch (int): Samples per encode call, or zero for the whole batch.

    Returns:
        tuple: ``(inputs, targets)``.

    Raises:
        ValueError: If no VAE was given.
    """
    device = _as_device(device)
    # With a latent cache the batch carries the latent and no video at all: `n` has to come from whichever of the
    # two the caller supplied.
    n = latents.shape[0] if latents is not None else batch["video"].shape[0]

    if latents is None and vae is None:
        raise ValueError(
            "pack_batch needs either a VAE to encode the video or a `latents` batch from a pre-extracted cache."
        )

    # The latent is always encoded here, never read from disk, and there is no way to hand one in. That removes
    # the only route by which a latent could reach the model without passing through normalize_latent, which is
    # worth more than the convenience: a cache written before that function existed holds latents off by a
    # per-channel affine map, relL2 0.52 against upstream, and training on them raises nothing and diverges from
    # nothing. A path that cannot be made safe is better absent than documented.
    #
    # Caching would be the wrong trade here anyway. Augmentation runs before the VAE, so a cache freezes it: six
    # augmented draws of one window sit 0.614 apart in relL2, further than two adjacent distinct windows (0.522),
    # so the cache would remove more variation than the dataset's own stride-1 sampling provides. And the recipe
    # runs 10000 iterations at a global batch of 8192 -- 4.86 passes over 16.3M windows -- so a lossless cache
    # would have to store every visit: 81.9M x 1.29 MiB = 100 TiB in bf16. Upstream's answer to the cost is to
    # make the online encode cheap instead: AOT-compiled chunk kernels, channels_last_3d weights, bf16.
    #
    # The VAE can only run in the main process, since workers cannot touch the device. Encode the batch in one
    # call and slice it per sample afterwards. The input follows the VAE's own dtype rather than a fixed one,
    # so that --vae-dtype actually changes what the convolutions compute instead of being upcast right here.
    if vae_encode_batch < 0:
        raise ValueError("vae_encode_batch must be nonnegative")
    if n < 1:
        raise ValueError("Cannot pack an empty video batch")
    with torch.no_grad():
        if latents is not None:
            # The cache holds latents already normalised and cropped to the content region, which is exactly what
            # pack_one takes. Re-normalising or re-cropping here would apply both twice.
            latent = latents.to(device).float()
            spatial_factor = None
        else:
            vae_dtype = next(vae.parameters()).dtype
            video = batch["video"]
            pad_hw = batch.get("video_pad_hw")
            uint8_video = video.dtype == torch.uint8

            def encode_video(video):
                if uint8_video:
                    # Same uint8 pixels and same affine as the host path; the float convert + reflect pad run on
                    # the device, so only 40 MB/sample crosses PCIe instead of 158 MB.
                    from examples.worldmodel.cosmos3.video_uint8 import pad_normalize_torch

                    target_h, target_w = (int(v) for v in pad_hw[0].tolist())
                    value = pad_normalize_torch(
                        video.to(device),
                        target_h - int(video.shape[-2]),
                        target_w - int(video.shape[-1]),
                        vae_dtype,
                    )
                else:
                    value = video.to(device, vae_dtype)
                out = vae.encode(value)
                dist = getattr(out, "latent_dist", None)
                if dist is not None:
                    return dist.mode()
                return out[0] if isinstance(out, (tuple, list)) else out

            encode = stopwatch.section("vae") if stopwatch is not None else contextlib.nullcontext()
            size = vae_encode_batch or n
            with encode:
                latent = torch.cat([encode_video(video[start : start + size]) for start in range(0, n, size)])
            # Micro-batching changes only encode calls. Normalization and its cast point match the baseline;
            # the clean payload remains FP32 until pack_one casts x_t at the model entrance.
            latent = normalize_latent(latent, vae).float()

    # The canvas is padded; the latent must not be. ``resize_and_reflect_pad`` fits the composite inside the
    # bucket canvas and reflection-pads the bottom and right edges, so for DROID a 540x640 composite sits in a
    # 544x736 frame. Upstream encodes the padded frame and then throws the padding away in latent space
    # (omni_mot_model.py::_remove_padding_from_latent, called on the encode's output): floor-divide the content
    # size by the spatial compression factor and keep that top-left corner, at least one row and column.
    #
    # Skipping this is not a rounding difference. It leaves 34x46 latent cells where upstream has 33x40, so the
    # vision stream carries 3519 tokens instead of 3060, the sequence length and every index after it move, and
    # 13% of the loss is computed on reflected pixels. Measured on episode 26827 offset 0.
    #
    # The factor is read off the two shapes rather than named as a constant, so a VAE whose compression differs
    # from the recipe's 16 cannot silently crop to the wrong place.
    if latents is None:
        if batch["video"].dtype == torch.uint8:
            # The uint8 batch carries the content; the padded target is what the device pad produced.
            target_h, target_w = (int(v) for v in batch["video_pad_hw"][0].tolist())
            spatial_factor = target_h // latent.shape[-2]
            compressed = (target_h, target_w)
        else:
            spatial_factor = batch["video"].shape[-2] // latent.shape[-2]
            compressed = tuple(batch["video"].shape[-2:])
        if latent.shape[-1] * spatial_factor != compressed[1]:
            raise ValueError(
                f"The VAE does not compress the two spatial axes equally: video {compressed} "
                f"-> latent {tuple(latent.shape[-2:])}; the padding cannot be mapped into latent space."
            )

    per_sample = []
    for index in range(n):
        # Action rows the prompt describes: the chunk plus the state row pack_one prepends.
        n_action_frames = int(batch["action"][index].shape[0]) + (1 if "state" in batch else 0)
        view_desc = batch.get("additional_view_description")
        idle = batch.get("idle_frames")
        prompt = build_action_json_prompt(
            batch["task"][index],
            num_frames=num_frames,
            fps=fps,
            canvas_hw=batch["canvas_hw"][index].tolist(),
            aspect=batch["aspect_key"][index],
            additional_view_description=view_desc[index] if view_desc is not None else None,
            idle_frames=int(idle[index]) if idle is not None else None,
            total_action_frames=n_action_frames,
        )
        ids = tokenize_prompt(prompt, tokenizer, generator=generator)
        content_h, content_w = (int(v) for v in batch["content_hw"][index].tolist())
        if latents is not None:
            # The cache entry is already cropped to the content region at extraction time; cropping again with the
            # same rule would be a no-op at best and a double crop if the geometry ever disagreed.
            piece = latent[index : index + 1].contiguous()
        else:
            keep_h = max(content_h // spatial_factor, 1)
            keep_w = max(content_w // spatial_factor, 1)
            piece = latent[index : index + 1, :, :, :keep_h, :keep_w].contiguous()
        per_sample.append(
            pack_one(
                piece,
                batch["action"][index].float(),
                ids,
                cfg,
                state=batch["state"][index].float() if "state" in batch else None,
                payload_dtype=payload_dtype,
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


def compute_loss(out, targets, return_components: bool = False):
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
        return_components (bool, optional): Also return the two weighted terms, detached, for logging. They
            sum to the scalar. Defaults to False.

    Returns:
        torch.Tensor or tuple: The scalar loss, or ``(loss, {"vision": term, "action": term})``.

    Raises:
        ValueError: If the model returned no vision or action predictions.
    """
    if not out.sample or not out.action:
        raise ValueError(
            "The model returned no vision or action predictions; check that the forward received non-empty "
            "vision and action inputs."
        )

    for modality, predictions in (("vision", out.sample), ("action", out.action)):
        if not (len(predictions) == len(targets[modality]) == len(targets[f"{modality}_noisy_mask"])):
            raise ValueError(f"{modality} predictions, targets and masks must have the same sample count")

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
    vision_term = vision * VISION_LOSS_SCALE
    action_term = action * ACTION_LOSS_WEIGHT
    total = vision_term + action_term
    if return_components:
        return total, {"vision": vision_term.detach(), "action": action_term.detach()}
    return total
