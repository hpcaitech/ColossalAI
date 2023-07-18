import argparse
import os
import socket
from functools import partial

import ray
import torch
from coati.quant import llama_load_quant, low_resource_init
from coati.ray.detached_trainer_ppo import DetachedPPOTrainer
from coati.ray.experience_maker_holder import ExperienceMakerHolder
from coati.ray.utils import (
    get_actor_from_args,
    get_critic_from_args,
    get_receivers_per_sender,
    get_reward_model_from_args,
    get_strategy_from_args,
)
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_utils import no_init_weights


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
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    tokenizer.pad_token = tokenizer.eos_token

    def model_fn():
        actor_cfg = AutoConfig.from_pretrained(args.pretrain)
        critic_cfg = AutoConfig.from_pretrained(args.critic_pretrain)
        actor = get_actor_from_args(args.model, config=actor_cfg).requires_grad_(False).half().cuda()
        critic = get_critic_from_args(args.critic_model, config=critic_cfg).requires_grad_(False).half().cuda()
        reward_model = get_reward_model_from_args(args.critic_model,
                                                  config=critic_cfg).requires_grad_(False).half().cuda()
        if args.initial_model_quant_ckpt is not None and args.model == 'llama':
            # quantize initial model
            with low_resource_init(), no_init_weights():
                initial_model = get_actor_from_args(args.model, config=actor_cfg)
            initial_model.model = llama_load_quant(initial_model.model, args.initial_model_quant_ckpt, args.quant_bits,
                                                   args.quant_group_size).cuda().requires_grad_(False)
        else:
            initial_model = get_actor_from_args(args.model, config=actor_cfg).requires_grad_(False).half().cuda()
        return actor, critic, reward_model, initial_model

    # configure Experience Maker
    experience_holder_ref = ExperienceMakerHolder.options(name="maker0", num_gpus=1, max_concurrency=2).remote(
        detached_trainer_name_list=[f'trainer{i}' for i in range(args.num_trainers)],
        strategy_fn=partial(get_strategy_from_args, args.maker_strategy),
        model_fn=model_fn,
        env_info=env_info_maker,
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

    def trainer_model_fn():
        actor = get_actor_from_args(args.model, config=AutoConfig.from_pretrained(args.pretrain)).half().cuda()
        critic = get_critic_from_args(args.critic_model,
                                      config=AutoConfig.from_pretrained(args.critic_pretrain)).half().cuda()
        return actor, critic

    # configure Trainer
    trainer_refs = [
        DetachedPPOTrainer.options(name=f"trainer{i}", num_gpus=1, max_concurrency=2).remote(
            experience_maker_holder_name_list=[
                f'maker{x}' for x in get_receivers_per_sender(i, args.num_trainers, 1, allow_idle_sender=True)
            ],
            strategy_fn=partial(get_strategy_from_args, args.trainer_strategy),
            model_fn=trainer_model_fn,
            env_info=env_info_trainer,
            train_batch_size=args.train_batch_size,
            buffer_limit=16,
            eval_performance=True,
            debug=args.debug,
        ) for i, env_info_trainer in enumerate(env_info_trainers)
    ]

    dataset_size = args.experience_batch_size * 4

    def data_gen_fn():
        input_ids = torch.randint(tokenizer.vocab_size, (256,), device=torch.cuda.current_device())
        attn_mask = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'attention_mask': attn_mask}

    def build_dataloader(size):
        dataset = [data_gen_fn() for _ in range(size)]
        dataloader = DataLoader(dataset, batch_size=args.experience_batch_size)
        return dataloader

    # uncomment this function if sync_models_from_trainers is True
    # ray.get([
    #     trainer_ref.sync_models_to_remote_makers.remote()
    #     for trainer_ref in trainer_refs
    # ])

    wait_tasks = []

    wait_tasks.append(
        experience_holder_ref.workingloop.remote(partial(build_dataloader, dataset_size),
                                                 num_steps=args.experience_steps))

    total_steps = args.experience_batch_size * args.experience_steps // (args.num_trainers * args.train_batch_size)
    for trainer_ref in trainer_refs:
        wait_tasks.append(trainer_ref.fit.remote(total_steps, args.update_steps, args.train_epochs))

    ray.get(wait_tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trainers', type=int, default=1)
    parser.add_argument('--trainer_strategy',
                        choices=[
                            'ddp', 'colossalai_gemini', 'colossalai_zero2', 'colossalai_gemini_cpu',
                            'colossalai_zero2_cpu'
                        ],
                        default='ddp')
    parser.add_argument('--maker_strategy', choices=['naive'], default='naive')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'llama'])
    parser.add_argument('--critic_model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'llama'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--critic_pretrain', type=str, default=None)
    parser.add_argument('--experience_steps', type=int, default=4)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--update_steps', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")

    parser.add_argument('--initial_model_quant_ckpt', type=str, default=None)
    parser.add_argument('--quant_bits', type=int, default=4)
    parser.add_argument('--quant_group_size', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    ray.init(namespace=os.environ["RAY_NAMESPACE"], runtime_env={"env_vars": dict(os.environ)})
    main(args)
