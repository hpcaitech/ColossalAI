"""DROID WAM inference with a diffusers transformer and ColossalAI attention.

Reuse the *official* CPU ActionTransformPipeline and FlowUniPC sampler. In particular,
CosmosActionCondition(mode='policy') alone isn't this recipe: DROID needs an initial
joint state, 8 raw channels, WAM video generation and an action time origin of zero.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch

CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite "
    "sides, with the robot visible."
)


def validate_observation(observation: dict, config: dict) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(observation, dict):
        raise ValueError("Observation must be a mapping")
    image = np.asarray(observation.get("observation/image"))
    shape = (config["image_height"], config["image_width"], 3)
    if image.shape != shape or image.dtype != np.uint8:
        raise ValueError(f"observation/image must be RGB uint8 with shape {shape}, got {image.shape}/{image.dtype}")
    joint = np.asarray(observation.get("observation/joint_position"), dtype=np.float32)
    gripper = np.asarray(observation.get("observation/gripper_position"), dtype=np.float32)
    if joint.shape not in ((7,), (1, 7)) or gripper.shape not in ((), (1,), (1, 1)):
        raise ValueError("Expected one joint_position [7] and one gripper_position [1]")
    if not np.isfinite(joint).all() or not np.isfinite(gripper).all() or not 0 <= gripper.item() <= 1:
        raise ValueError("Robot state must be finite and gripper_position must be in [0, 1]")
    prompt = observation.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty instruction string")
    # RoboLab is 1=open; DROID is 1=closed. The reference server flips both input and output.
    state = np.concatenate((joint.reshape(7), [1.0 - gripper.item()])).astype(np.float32)
    return np.ascontiguousarray(image), state, prompt


def build_sample(observation: dict, config: dict, transform) -> dict:
    image, state, prompt = validate_observation(observation, config)
    frames = config["chunk_size"] + 1
    video = torch.zeros(3, frames, *image.shape[:2], dtype=torch.uint8)
    video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)
    action = torch.zeros(frames, config["action_dim"], dtype=torch.float32)
    action[0] = torch.from_numpy(state)
    return transform(
        {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(config["fps"], dtype=torch.long),
            "mode": "wam",
            "domain_id": torch.tensor(config["domain_id"], dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": CONCAT_VIEW_DESCRIPTION,
        },
        config["resolution"],
    )


def _text_ids(tokenizer, prompt: dict | str, device) -> torch.Tensor:
    if isinstance(prompt, dict):
        prompt = json.dumps(prompt)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=False,
    )
    return torch.tensor(ids[:4096], dtype=torch.long, device=device)


def pack_inference_inputs(vision: torch.Tensor, action: torch.Tensor, ids: torch.Tensor, config, fps: float) -> dict:
    """One clean vision frame and one clean state row; all future tokens share a timestep.

    mRoPE follows sequence_packing/mrope.py. With use_state=True, action index zero
    represents time zero, and action index 4 aligns with latent vision frame 1.
    """
    device = vision.device
    _, _, frames, height, width = vision.shape
    patch = config.latent_patch_size
    if height % patch or width % patch or action.shape[1] != config.action_dim:
        raise ValueError("Latent geometry/action width does not match the transformer")
    hp, wp = height // patch, width // patch
    spatial = hp * wp
    text_length, action_length = ids.numel(), action.shape[0]
    vision_length = frames * spatial
    length = text_length + vision_length + action_length
    text_pos = torch.arange(text_length, device=device, dtype=torch.float32).expand(3, -1)
    offset = text_length + config.unified_3d_mrope_temporal_modality_margin
    base_fps = config.base_fps
    # DROID's Wan VAE compresses time by 4; both streams share the vision temporal unit.
    base_tps = base_fps / 4
    vision_t = torch.arange(frames, device=device, dtype=torch.float32) / (fps / 4) * base_tps + offset
    vision_pos = torch.stack(
        [
            vision_t.repeat_interleave(spatial),
            torch.arange(hp, device=device).repeat_interleave(wp).repeat(frames),
            torch.arange(wp, device=device).repeat(frames * hp),
        ]
    )
    action_t = torch.arange(action_length, device=device, dtype=torch.float32) / fps * base_tps + offset
    action_pos = torch.stack([action_t, torch.zeros_like(action_t), torch.zeros_like(action_t)])
    vis_idx = torch.arange(text_length, text_length + vision_length, device=device)
    act_idx = torch.arange(text_length + vision_length, length, device=device)
    return {
        "input_ids": ids,
        "text_indexes": torch.arange(text_length, device=device),
        "position_ids": torch.cat([text_pos, vision_pos, action_pos], dim=1),
        "und_len": text_length,
        "sequence_length": length,
        "vision_tokens": [vision],
        "vision_token_shapes": [(frames, hp, wp)],
        "vision_sequence_indexes": vis_idx,
        "vision_mse_loss_indexes": vis_idx[spatial:],
        "vision_noisy_frame_indexes": [torch.arange(1, frames, device=device)],
        "action_tokens": [action],
        "action_token_shapes": [(action_length, 1, 1)],
        "action_sequence_indexes": act_idx,
        "action_mse_loss_indexes": act_idx[1:],
        "action_noisy_frame_indexes": [torch.arange(1, action_length, device=device)],
        "action_domain_ids": [torch.tensor(8, device=device)],
        "return_dict": False,
    }


def format_actions(action: torch.Tensor, config: dict) -> np.ndarray:
    if action.ndim != 2 or action.shape[0] != config["chunk_size"] + 1 or action.shape[1] < config["action_dim"]:
        raise ValueError(f"Expected state + {config['chunk_size']} predicted joint actions, got {tuple(action.shape)}")
    result = action[1:, : config["action_dim"]].detach().float().cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise RuntimeError("Policy generated nonfinite actions")
    result[:, -1] = 1.0 - result[:, -1]
    # Thresholding > 0.5 is owned by the official Cosmos3Client, after the wire conversion.
    return result


def initial_noise(shape, seed: int, device) -> torch.Tensor:
    """Match official misc.arch_invariant_rand, restarting the seed for each modality."""
    values = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(values).to(device=device)


class DroidWamPolicy:
    def __init__(self, transformer, vae, tokenizer, transform, scheduler_factory, config: dict, device):
        self.transformer, self.vae, self.tokenizer = transformer, vae, tokenizer
        self.transform, self.scheduler_factory = transform, scheduler_factory
        self.config, self.device = config, torch.device(device)

    @classmethod
    def from_pretrained(
        cls, checkpoint: Path, *, transformer_path: Path | None, config: dict, device: str, backend: str
    ):
        if device.startswith("npu"):
            importlib.import_module("torch_npu")
        resolved = torch.device(device)
        from examples.worldmodel.cosmos3.runtime import resolve_attention_backend

        backend = resolve_attention_backend(backend, resolved.type)
        if resolved.type not in ("cpu", "cuda", "npu"):
            raise ValueError("Policy inference supports CPU, CUDA or NPU devices")

        from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
        from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer
        from transformers import AutoTokenizer

        from colossalai.shardformer.modeling.cosmos3 import (
            ColossalTwoWayAttnProcessor,
            install_fp32_sinusoidal_timestep_input,
            install_fp32_timestep_embedding,
        )

        if resolved.type == "npu":
            torch.npu.set_device(resolved)
            torch.npu.conv.allow_hf32 = False
            torch.npu.matmul.allow_hf32 = False
        elif resolved.type == "cuda":
            torch.cuda.set_device(resolved)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        transformer = (
            Cosmos3OmniTransformer.from_pretrained(
                str(transformer_path or checkpoint / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
            .to(resolved)
        )
        install_fp32_timestep_embedding(transformer)
        install_fp32_sinusoidal_timestep_input(transformer)
        if not transformer.config.action_gen or transformer.config.action_dim < config["action_dim"]:
            raise ValueError("Checkpoint must enable the DROID action generation head")
        if not transformer.config.enable_fps_modulation or not transformer.config.unified_3d_mrope_reset_spatial_ids:
            raise ValueError("DROID WAM expects FPS modulation and spatial mRoPE indices reset per modality")
        processor, installed = ColossalTwoWayAttnProcessor(backend=backend), 0
        for module in transformer.modules():
            if module.__class__.__name__ == "Cosmos3PackedMoTAttention":
                module.set_processor(processor)
                installed += 1
        if not installed:
            raise ValueError("Checkpoint has no Cosmos3 packed MoT attention modules")
        vae = (
            AutoencoderKLWan.from_pretrained(str(checkpoint / "vae"), torch_dtype=torch.float32, local_files_only=True)
            .eval()
            .requires_grad_(False)
            .to(resolved)
        )
        if vae.config.scale_factor_temporal != 4:
            raise ValueError("DROID WAM expects a Wan VAE with temporal compression 4")
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / "text_tokenizer"), local_files_only=True)
        transform = ActionTransformPipeline(
            max_action_dim=transformer.config.action_dim, cfg_dropout_rate=0.0, format_prompt_as_json=True
        )
        return cls(transformer, vae, tokenizer, transform, FlowUniPCMultistepScheduler, config, resolved)

    @torch.inference_mode()
    def infer(self, observation: dict) -> dict:
        # websockets.sync uses a thread per connection; accelerator selection is thread-local.
        if self.device.type in ("cuda", "npu"):
            getattr(torch, self.device.type).set_device(self.device)
        sample = build_sample(observation, self.config, self.transform)
        pixels = sample["video"][:, :1].unsqueeze(0).to(self.device, torch.float32) / 127.5 - 1.0
        first = self.vae.encode(pixels).latent_dist.mode().float()
        # Wan's native encoder normalizes its posterior mean; diffusers exposes the raw mean.
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=self.device).view(1, -1, 1, 1, 1)
        first = (first - mean) / std
        frames = self.config["chunk_size"] // 4 + 1
        vision = initial_noise((1, first.shape[1], frames, *first.shape[-2:]), self.config["seed"], self.device)
        action = initial_noise(sample["action"].shape, self.config["seed"], self.device)
        state = sample["action"][0].to(self.device)
        vision[:, :, :1], action[0] = first, state
        action[:, self.config["action_dim"] :] = 0
        ids = [_text_ids(self.tokenizer, sample["ai_caption"], self.device)]
        if self.config["guidance"] != 1.0:
            ids.append(_text_ids(self.tokenizer, "", self.device))
        # The native solver carries history. Each modality and each request needs a fresh instance.
        schedulers = [self.scheduler_factory(shift=1.0) for _ in range(2)]
        for scheduler in schedulers:
            scheduler.set_timesteps(self.config["num_steps"], device=self.device, shift=self.config["shift"])
        for timestep in schedulers[0].timesteps:
            predictions = []
            for branch_ids in ids:
                inputs = pack_inference_inputs(
                    vision.to(torch.bfloat16),
                    action.to(torch.bfloat16),
                    branch_ids,
                    self.transformer.config,
                    self.config["fps"],
                )
                inputs["vision_timesteps"] = torch.full(
                    (len(inputs["vision_mse_loss_indexes"]),), timestep.item(), device=self.device, dtype=torch.float32
                )
                inputs["action_timesteps"] = torch.full(
                    (action.shape[0] - 1,), timestep.item(), device=self.device, dtype=torch.float32
                )
                pred_vision, _, pred_action = self.transformer(**inputs)
                predictions.append((pred_vision[0].float(), pred_action[0].float()))
            velocity_vision, velocity_action = predictions[0]
            if len(predictions) == 2:
                uncond_vision, uncond_action = predictions[1]
                velocity_vision = uncond_vision + self.config["guidance"] * (velocity_vision - uncond_vision)
                velocity_action = uncond_action + self.config["guidance"] * (velocity_action - uncond_action)
            velocity_vision[:, :, :1] = 0
            velocity_action[0] = 0
            velocity_action[:, self.config["action_dim"] :] = 0
            vision = schedulers[0].step(velocity_vision, timestep, vision, return_dict=False)[0]
            action = schedulers[1].step(velocity_action.unsqueeze(0), timestep, action.unsqueeze(0), return_dict=False)[
                0
            ][0]
            vision[:, :, :1], action[0] = first, state
            action[:, self.config["action_dim"] :] = 0
        return {"action": format_actions(action, self.config)}
