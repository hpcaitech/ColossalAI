import argparse
import os
import socket
from functools import partial

import ray
import torch
from coati.ray.src.detached_trainer_ppo import DetachedPPOTrainer
from coati.ray.src.experience_maker_holder import ExperienceMakerHolder
from coati.ray.src.utils import (
    get_actor_from_args,
    get_critic_from_args,
    get_reward_model_from_args,
    get_strategy_from_args,
)
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer


def get_gpt_config(model_name: str) -> GPT2Config:
    model_map = {
        's': GPT2Config(),
        'm': GPT2Config(n_embd=1024, n_layer=24, n_head=16),
        'l': GPT2Config(n_embd=1280, n_layer=36, n_head=20),
        'xl': GPT2Config(n_embd=1600, n_layer=48, n_head=25),
        '2b': GPT2Config(n_embd=2048, n_layer=40, n_head=16),
        '4b': GPT2Config(n_embd=2304, n_layer=64, n_head=16),
        '6b': GPT2Config(n_embd=4096, n_layer=30, n_head=16),
        '8b': GPT2Config(n_embd=4096, n_layer=40, n_head=16),
        '10b': GPT2Config(n_embd=4096, n_layer=50, n_head=16),
        '12b': GPT2Config(n_embd=4096, n_layer=60, n_head=16),
        '15b': GPT2Config(n_embd=4096, n_layer=78, n_head=16),
        '18b': GPT2Config(n_embd=4096, n_layer=90, n_head=16),
        '20b': GPT2Config(n_embd=8192, n_layer=25, n_head=16),
        '24b': GPT2Config(n_embd=8192, n_layer=30, n_head=16),
        '28b': GPT2Config(n_embd=8192, n_layer=35, n_head=16),
        '32b': GPT2Config(n_embd=8192, n_layer=40, n_head=16),
        '36b': GPT2Config(n_embd=8192, n_layer=45, n_head=16),
        '40b': GPT2Config(n_embd=8192, n_layer=50, n_head=16),
        '175b': GPT2Config(n_positions=2048, n_embd=12288, n_layer=96, n_head=96),
    }
    try:
        return model_map[model_name]
    except KeyError:
        raise ValueError(f'Unknown model "{model_name}"')


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]


def main(args):
    master_addr = str(get_local_ip())
    # trainer_env_info
    trainer_port = str(get_free_port())
    env_info_trainers = [{
        'local_rank': '0',
        'rank': str(rank),
        'world_size': str(args.num_trainers),
        'master_port': trainer_port,
        'master_addr': master_addr
    } for rank in range(args.num_trainers)]

    # maker_env_info
    maker_port = str(get_free_port())
    env_info_maker = {
        'local_rank': '0',
        'rank': '0',
        'world_size': '1',
        'master_port': maker_port,
        'master_addr': master_addr
    }

    # configure tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    def trainer_model_fn():
        actor = get_actor_from_args(args.model, args.pretrain).half().cuda()
        critic = get_critic_from_args(args.model, args.pretrain).half().cuda()
        return actor, critic

    # configure Trainer
    trainer_refs = [
        DetachedPPOTrainer.options(name=f"trainer{i}", num_gpus=1, max_concurrency=2).remote(
            experience_maker_holder_name_list=["maker1"],
            strategy_fn=partial(get_strategy_from_args, args.trainer_strategy),
            model_fn=trainer_model_fn,
            env_info=env_info_trainer,
            train_batch_size=args.train_batch_size,
            buffer_limit=16,
            max_epochs=args.max_epochs,
            eval_performance=True,
            debug=args.debug,
        ) for i, env_info_trainer in enumerate(env_info_trainers)
    ]

    def model_fn():
        actor = get_actor_from_args(args.model, args.pretrain).half().cuda()
        critic = get_critic_from_args(args.model, args.pretrain).half().cuda()
        reward_model = get_reward_model_from_args(args.model, args.pretrain).half().cuda()
        initial_model = get_actor_from_args(args.model, args.pretrain).half().cuda()
        return actor, critic, reward_model, initial_model

    # configure Experience Maker
    experience_holder_ref = ExperienceMakerHolder.options(name="maker1", num_gpus=1, max_concurrency=2).remote(
        detached_trainer_name_list=[f'trainer{i}' for i in range(args.num_trainers)],
        strategy_fn=partial(get_strategy_from_args, args.maker_strategy),
        model_fn=model_fn,
        env_info=env_info_maker,
        experience_batch_size=args.experience_batch_size,
        kl_coef=0.1,
        debug=args.debug,
    # sync_models_from_trainers=True,
    # generation kwargs:
        max_length=512,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        eval_performance=True,
        use_cache=True,
    )

    # configure sampler
    random_prompts = torch.randint(tokenizer.vocab_size, (1000, 400))

    def tokenize_fn(texts):
        input_ids = torch.stack(texts).cuda()
        attn_mask = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'attention_mask': attn_mask}

    # uncomment this function if sync_models_from_trainers is True
    # ray.get([
    #     trainer_ref.sync_models_to_remote_makers.remote()
    #     for trainer_ref in trainer_refs
    # ])

    wait_tasks = []

    for trainer_ref in trainer_refs:
        wait_tasks.append(
            trainer_ref.fit.remote(num_episodes=args.num_episodes,
                                   max_timesteps=args.max_timesteps,
                                   update_timesteps=args.update_timesteps))

    num_exp_per_maker = args.num_episodes * args.max_timesteps // args.update_timesteps * \
        args.max_epochs * args.num_trainers + 3  # +3 for fault tolerance
    wait_tasks.append(experience_holder_ref.workingloop.remote(random_prompts, tokenize_fn, times=num_exp_per_maker))

    ray.get(wait_tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trainers', type=int, default=1)
    parser.add_argument('--trainer_strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--maker_strategy', choices=['naive'], default='naive')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    ray.init(namespace=os.environ["RAY_NAMESPACE"])
    main(args)
