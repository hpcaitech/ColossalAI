# import copy
# import os

# import pytest
# import torch
# import torch.distributed as dist
# from contextlib import nullcontext
# from transformers.models.gpt2.configuration_gpt2 import GPT2Config
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from coati.experience_buffer import NaiveExperienceBuffer
# from coati.experience_maker import NaiveExperienceMaker
# import colossalai
# from coati.models import RewardModel, Critic
# from colossalai.booster import Booster
# from colossalai.lazy import LazyInitContext
# from colossalai.utils import get_current_device
# from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
# from colossalai.testing import rerun_if_address_is_in_use, spawn


# def get_data(batch_size: int, seq_len: int = 10) -> dict:
#     input_ids = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
#     attention_mask = torch.ones_like(input_ids)
#     return dict(input_ids=input_ids, attention_mask=attention_mask)


# def gather_and_equal(tensor: torch.Tensor) -> bool:
#     world_size = dist.get_world_size()
#     outputs = [torch.empty_like(tensor) for _ in range(world_size)]
#     dist.all_gather(outputs, tensor.contiguous())
#     for t in outputs[1:]:
#         if not torch.equal(outputs[0], t):
#             return False
#     return True


# def make_and_consume_experience(plugin_type, mixed_precision, tp):

#     colossalai.launch_from_torch({})

#     EXPERIENCE_BATCH_SIZE = 4
#     SAMPLE_BATCH_SIZE = 2
#     custom_plugin = None

#     if plugin_type == "gemini":
#         plugin = GeminiPlugin(
#             precision=mixed_precision,
#             initial_scale=2**16,
#             max_norm=1.0,
#         )
#     elif plugin_type == "gemini_auto":
#         plugin = GeminiPlugin(
#             precision=mixed_precision,
#             placement_policy="auto",
#             initial_scale=2**16,
#             max_norm=1.0,
#         )
#     elif plugin_type == "zero2":
#         plugin = LowLevelZeroPlugin(
#             stage=2,
#             precision=mixed_precision,
#             initial_scale=2**16,
#             max_norm=1.0,
#         )
#     elif plugin_type == "zero2_cpu":
#         plugin = LowLevelZeroPlugin(
#             stage=2,
#             precision=mixed_precision,
#             initial_scale=2**16,
#             cpu_offload=True,
#             max_norm=1.0,
#         )
#     elif plugin_type == "3d":
#         plugin = HybridParallelPlugin(
#             tp_size=tp,
#             pp_size=1,
#             zero_stage=0,
#             precision=mixed_precision,
#         )
#         from colossalai.shardformer.policies.gpt2 import GPT2Policy
#         custom_plugin = HybridParallelPlugin(
#             tp_size=tp,
#             pp_size=1,
#             zero_stage=0,
#             precision=mixed_precision,
#             custom_policy=GPT2Policy(),
#         )
#     else:
#         raise ValueError(f'Unsupported plugin "{plugin}"')

#     # init_ctx = LazyInitContext(default_device=get_current_device()) if "gemini" in plugin_type else nullcontext()
#     # with init_ctx:
#     actor = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
#     critic = Critic("gpt2").cuda()

#     ref_model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
#     reward_model = RewardModel("gpt2").cuda()

#     actor_booster = Booster(plugin=plugin)
#     ref_booster = Booster(plugin=plugin)
#     rm_booster = Booster(plugin=custom_plugin)
#     critic_booster = Booster(plugin=custom_plugin)

#     default_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
#     torch.set_default_dtype(default_dtype)
#     actor, _, _, _, _ = actor_booster.boost(
#         model=actor
#     )

#     critic, _, _, _, _ = critic_booster.boost(
#         model=critic
#     )
#     reward_model, _, _, _, _ = rm_booster.boost(model=reward_model)
#     ref_model, _, _, _, _ = ref_booster.boost(model=ref_model)

#     torch.set_default_dtype(torch.float)

#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     tokenizer.padding_side = "left"
#     tokenizer.pad_token = tokenizer.eos_token
#     experience_maker = NaiveExperienceMaker(actor, critic, reward_model, ref_model, tokenizer)
#     data_buffer = NaiveExperienceBuffer(SAMPLE_BATCH_SIZE, cpu_offload=False)

#     # experience of all ranks should be the same
#     for _ in range(2):
#         data = get_data(EXPERIENCE_BATCH_SIZE)
#         assert gather_and_equal(data["input_ids"])
#         assert gather_and_equal(data["attention_mask"])
#         experience = experience_maker.make_experience(**data, do_sample=True, max_length=16)
#         assert gather_and_equal(experience.sequences)
#         assert gather_and_equal(experience.action_log_probs)
#         assert gather_and_equal(experience.values)
#         assert gather_and_equal(experience.reward)
#         assert gather_and_equal(experience.advantages)
#         assert gather_and_equal(experience.action_mask)
#         assert gather_and_equal(experience.attention_mask)
#         data_buffer.append(experience)

#     # data buffer's data should be the same for tp but different for other methods
#     # buffer_size = torch.tensor([len(data_buffer)], device="cuda")
#     # assert gather_and_equal(buffer_size)
#     # for item in data_buffer.items:
#     #     assert gather_and_equal(item.sequences)
#     #     assert gather_and_equal(item.action_log_probs)
#     #     assert gather_and_equal(item.values)
#     #     assert gather_and_equal(item.reward)
#     #     assert gather_and_equal(item.advantages)
#     #     assert gather_and_equal(item.action_mask)
#     #     assert gather_and_equal(item.attention_mask)

#     # # dataloader of each rank should have the same size and different batch
#     # dataloader = strategy.setup_dataloader(data_buffer)
#     # dataloader_size = torch.tensor([len(dataloader)], device="cuda")
#     # assert gather_and_equal(dataloader_size)
#     # for experience in dataloader:
#     #     assert not gather_and_equal(experience.sequences)
#     #     assert not gather_and_equal(experience.action_log_probs)
#     #     assert not gather_and_equal(experience.values)
#     #     assert not gather_and_equal(experience.reward)
#     #     assert not gather_and_equal(experience.advantages)
#     #     # action mask and attention mask may be same


# def run_dist(rank, world_size, port, plugin=None, mixed_precision=None, tp=None):
#     os.environ["RANK"] = str(rank)
#     os.environ["LOCAL_RANK"] = str(rank)
#     os.environ["WORLD_SIZE"] = str(world_size)
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(port)
#     make_and_consume_experience(plugin, mixed_precision, tp)


# @pytest.mark.dist
# @pytest.mark.parametrize("world_size", [2])
# @pytest.mark.parametrize("plugin", ["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d"])
# @pytest.mark.parametrize("mixed_precision", ["fp16", "bf16"])
# @pytest.mark.parametrize("tp", [2]) #only for 3d plugin
# @rerun_if_address_is_in_use()
# def test_experience(world_size, plugin, mixed_precision, tp):
#     spawn(run_dist, world_size, plugin=plugin, mixed_precision=mixed_precision, tp=tp)


# if __name__ == "__main__":
#     test_experience(2, "colossalai-zero2")
