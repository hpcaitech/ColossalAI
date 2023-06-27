import argparse
import logging
import os
import socket
from copy import deepcopy
from typing import Type

import ray
import torch
from coati.experience_maker.base import Experience
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMActor, BLOOMCritic
from coati.models.gpt import GPTActor, GPTCritic
from coati.models.lora import LoRAModule
from coati.models.loss import PolicyLoss, ValueLoss
from coati.models.opt import OPTActor, OPTCritic
from coati.models.utils import compute_reward
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam


class ExperienceCompositionRefs:

    def __init__(self, sequences_attention_mask_action_mask_ref: ray.ObjectRef, action_log_probs_ref: ray.ObjectRef,
                 base_action_log_probs_ref: ray.ObjectRef, value_ref: ray.ObjectRef, r_ref: ray.ObjectRef) -> None:
        self.sequences_attention_mask_action_mask_ref = sequences_attention_mask_action_mask_ref
        self.action_log_probs_ref = action_log_probs_ref
        self.base_action_log_probs_ref = base_action_log_probs_ref
        self.value_ref = value_ref
        self.r_ref = r_ref


class ExperienceMaker:

    def __init__(self, kl_coef) -> None:
        self.kl_coef = kl_coef

    @torch.no_grad()
    def make_experience(self, experiment_computation_refs: ExperienceCompositionRefs):
        sequences, attention_mask, action_mask = ray.get(
            experiment_computation_refs.sequences_attention_mask_action_mask_ref)
        action_log_probs = ray.get(experiment_computation_refs.action_log_probs_ref)
        base_action_log_probs = ray.get(experiment_computation_refs.base_action_log_probs_ref)
        r = ray.get(experiment_computation_refs.r_ref)
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)
        value = ray.get(experiment_computation_refs.value_ref)
        advantage = reward - value
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)
        experience = Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
        return experience


class DistributedTorchRayActor:

    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        self._model = None
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(self._local_rank)

    @staticmethod
    def _get_current_node_ip():
        return ray._private.services.get_node_ip_address()

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(('', 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BasePPORole(DistributedTorchRayActor):

    def add_experience_maker(self, kl_coef: float = 0.1):
        self._experience_maker = ExperienceMaker(kl_coef)

    def make_experience(self, experience_computation_ref: ExperienceCompositionRefs):
        return self._experience_maker.make_experience(experience_computation_ref)

    def _init_strategy(self, strategy: str):
        # configure strategy
        if strategy == 'ddp':
            self._strategy = DDPStrategy()
        elif strategy == 'colossalai_gemini':
            self._strategy = GeminiStrategy(placement_policy='cuda', initial_scale=2**5)
        elif strategy == 'colossalai_zero2':
            self._strategy = LowLevelZeroStrategy(stage=2, placement_policy='cuda')
        else:
            raise ValueError(f'Unsupported strategy "{strategy}"')

    def _init_optimizer(self):
        if isinstance(self._strategy, (GeminiStrategy, LowLevelZeroStrategy)):
            self._optimizer = HybridAdam(self._model.parameters(), lr=5e-6)
        else:
            self._optimizer = Adam(self._model.parameters(), lr=5e-6)

    def _prepare_model_with_strategy(self, has_optimizer: bool):
        if has_optimizer:
            self._init_optimizer()
            (self._model, self._optimizer) = self._strategy.prepare((self._model, self._optimizer))
        else:
            self._model = self._strategy.prepare(self._model)

    def _load_model_from_pretrained(self, model_class: Type[LoRAModule], pretrain: str):
        raise NotImplementedError()

    def init_model_from_pretrained(self,
                                   strategy: str,
                                   model_class: Type[LoRAModule],
                                   pretrain: str,
                                   has_optimizer=False):
        self._init_strategy(strategy)
        self._load_model_from_pretrained(model_class, pretrain)
        self._prepare_model_with_strategy(has_optimizer)

    def eval(self):
        self._model.eval()


class TrainablePPORole(BasePPORole):

    def _load_model_from_pretrained(self, model_class, pretrain):
        with self._strategy.model_init_context():
            self._model = model_class(pretrain).to(torch.cuda.current_device())

    def _train(self):
        self._model.train()

    def _training_step(self, experience: Experience):
        raise NotImplementedError()

    def learn_on_experiences(self, experience_refs):
        experiences = ray.get(experience_refs)
        device = torch.cuda.current_device()
        self._train()
        for exp in experiences:
            exp.to_device(device)
            self._training_step(exp)
        self.eval()


@ray.remote(num_gpus=1)
class RayPPOActor(TrainablePPORole):

    def set_loss_function(self, eps_clip: float):
        self._actor_loss_fn = PolicyLoss(eps_clip)

    def load_tokenizer_from_pretrained(self, model_type: str, pretrained):
        if model_type == 'gpt2':
            self._model_tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
            self._model_tokenizer.pad_token = self._model_tokenizer.eos_token
        elif model_type == 'bloom':
            self._model_tokenizer = BloomTokenizerFast.from_pretrained(pretrained)
            self._model_tokenizer.pad_token = self._model_tokenizer.eos_token
        elif model_type == 'opt':
            self._model_tokenizer = AutoTokenizer.from_pretrained(pretrained)
        else:
            raise ValueError(f'Unsupported model "{model_type}"')

        # Set tokenize function for sequence generation
        def _text_input_tokenize_fn(texts):
            batch = self._model_tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)
            return {k: v.cuda() for k, v in batch.items()}

        self._sample_tokenize_function = _text_input_tokenize_fn

    def setup_generate_kwargs(self, generate_kwargs: dict):
        from coati.trainer.ppo import _set_default_generate_kwargs
        self._generate_kwargs = _set_default_generate_kwargs(self._strategy, generate_kwargs, self._model)
        self._generate_kwargs['pad_token_id'] = self._model_tokenizer.pad_token_id
        self._generate_kwargs['eos_token_id'] = self._model_tokenizer.eos_token_id

    def load_csv_prompt_file_from_url_to_sampler(self, prompt_url):
        import pandas as pd
        prompts = pd.read_csv(prompt_url)['prompt']
        self._sampler = self._strategy.setup_sampler(prompts)

    def _generate(self, input_ids, **generate_kwargs):
        return self._model.generate(input_ids, return_action_mask=True, **generate_kwargs)

    def sample_prompts_and_make_sequence(self, experience_batch_size):
        sampled_prompts = self._sampler.sample(experience_batch_size)
        input_ids = self._sample_tokenize_function(sampled_prompts)
        if isinstance(input_ids, dict):
            return self._generate(**input_ids, **self._generate_kwargs)
        else:
            return self._generate(input_ids, **self._generate_kwargs)

    @torch.no_grad()
    def calculate_action_log_probs(self, sequence_attention_action_mask):
        sequences, attention_mask, action_mask = sequence_attention_action_mask
        return self._model.forward(sequences, action_mask.size(1), attention_mask)

    def _training_step(self, experience):
        num_actions = experience.action_mask.size(1)
        action_log_probs = self._model(experience.sequences, num_actions, attention_mask=experience.attention_mask)
        actor_loss = self._actor_loss_fn(action_log_probs,
                                         experience.action_log_probs,
                                         experience.advantages,
                                         action_mask=experience.action_mask)
        self._strategy.backward(actor_loss, self._model, self._optimizer)
        self._strategy.optimizer_step(self._optimizer)
        self._optimizer.zero_grad()
        logging.info("actor_loss: {}".format(actor_loss))

    def save_checkpoint(self, save_path, should_save_optimizer: bool):
        if self._rank == 0:
            # save model checkpoint only on rank 0
            self._strategy.save_model(self._model, save_path, only_rank0=True)
        # save optimizer checkpoint on all ranks
        if should_save_optimizer:
            self._strategy.save_optimizer(self._optimizer,
                                          'actor_optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
                                          only_rank0=False)

    def generate_answer(self, prompt, max_length=30, num_return_sequences=5):
        encoded_input = self._model_tokenizer(prompt, return_tensors='pt')
        input_ids = {k: v.cuda() for k, v in encoded_input.items()}
        sequence, _ = self._model.generate(**input_ids,
                                           max_length=max_length,
                                           return_action_mask=False,
                                           num_return_sequences=num_return_sequences)
        token_list = list(sequence.data[0])
        output = " ".join([self._model_tokenizer.decode(token) for token in token_list])
        return output


@ray.remote(num_gpus=1)
class RayPPOCritic(TrainablePPORole):

    def set_loss_function(self, value_clip: float):
        self._critic_loss_fn = ValueLoss(value_clip)

    def _training_step(self, experience):
        values = self._model(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask)
        critic_loss = self._critic_loss_fn(values,
                                           experience.values,
                                           experience.reward,
                                           action_mask=experience.action_mask)
        self._strategy.backward(critic_loss, self._model, self._optimizer)
        self._strategy.optimizer_step(self._optimizer)
        self._optimizer.zero_grad()
        logging.info("critic_loss: {}".format(critic_loss))

    @torch.no_grad()
    def calculate_value(self, sequence_attention_action_mask):
        sequences, attention_mask, action_mask = sequence_attention_action_mask
        return self._model(sequences, action_mask, attention_mask)


@ray.remote(num_gpus=1)
class RayPPORewardModel(BasePPORole):

    def _load_model_from_pretrained(self, model_class, pretrain):
        with self._strategy.model_init_context():
            critic = model_class(pretrained=pretrain).to(torch.cuda.current_device())
            self._model = RewardModel(deepcopy(critic.model),
                                      deepcopy(critic.value_head)).to(torch.cuda.current_device())

    @torch.no_grad()
    def calculate_r(self, sequence_attention_action_mask):
        sequences, attention_mask, _ = sequence_attention_action_mask
        return self._model(sequences, attention_mask)


@ray.remote(num_gpus=1)
class RayPPOInitialModel(BasePPORole):

    def _load_model_from_pretrained(self, model_class, pretrain):
        with self._strategy.model_init_context():
            self._model = model_class(pretrain).to(torch.cuda.current_device())

    @torch.no_grad()
    def calculate_base_action_log_probs(self, sequence_attention_action_mask):
        sequences, attention_mask, action_mask = sequence_attention_action_mask
        return self._model(sequences, action_mask.size(1), attention_mask)


class PPORayActorGroup:
    """
        A group of ray actors
        Functions start with 'async' should return list of object refs
    """

    def __init__(self, num_nodes, num_gpus_per_node, ray_actor_type: Type[BasePPORole]) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        self._initiate_actors()

    def _initiate_actors(self):
        world_size = self._num_nodes * self._num_gpus_per_node
        # Use placement group to lock resources for models of same type
        pg = None
        if self._num_gpus_per_node > 1:
            bundles = [{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0)).remote(world_size, 0, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(num_gpus=1).remote(world_size, 0, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    worker_actor = self.ray_actor_type.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=rank // self._num_gpus_per_node)).remote(
                            world_size, rank, local_rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(num_gpus=1).remote(world_size, rank, local_rank,
                                                                                  master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(self, strategy: str, model_class: Type[LoRAModule], pretrain: str,
                                         has_optimizer: bool):
        return [
            actor.init_model_from_pretrained.remote(strategy, model_class, pretrain, has_optimizer)
            for actor in self._actor_handlers
        ]


class TrainableModelRayActorGroup(PPORayActorGroup):

    def async_learn_on_experiences(self, experience_refs):
        num_actors = len(self._actor_handlers)
        learn_result_refs = []
        for i in range(num_actors):
            exp_refs_batch = experience_refs[i::num_actors]
            learn_result_refs.append(self._actor_handlers[i].learn_on_experiences.remote(exp_refs_batch))
        return learn_result_refs


class PPOActorRayActorGroup(TrainableModelRayActorGroup):

    def __init__(self, num_nodes, num_gpus_per_node) -> None:
        super().__init__(num_nodes, num_gpus_per_node, RayPPOActor)

    def async_prepare_for_sequence_generation(self, model: str, pretrain: str, generation_kwargs: dict):
        refs = []
        for actor in self._actor_handlers:
            refs.append(actor.load_tokenizer_from_pretrained.remote(model, pretrain))
            refs.append(actor.setup_generate_kwargs.remote(generation_kwargs))
        return refs

    def load_csv_prompt_file_from_url_to_sampler(self, csv_url):
        ray.get([actor.load_csv_prompt_file_from_url_to_sampler.remote(csv_url) for actor in self._actor_handlers])

    def async_sample_prompts_and_make_sequence(self, experience_batch_size):
        return [actor.sample_prompts_and_make_sequence.remote(experience_batch_size) for actor in self._actor_handlers]

    def async_calculate_action_log_probs(self, sequences_attention_mask_action_mask_refs):
        num_actors = len(self._actor_handlers)
        action_log_probs_refs = []
        for i in range(len(sequences_attention_mask_action_mask_refs)):
            action_log_probs_ref = self._actor_handlers[i % num_actors].calculate_action_log_probs.remote(
                sequences_attention_mask_action_mask_refs[i])
            action_log_probs_refs.append(action_log_probs_ref)
        return action_log_probs_refs

    def set_loss_function(self, eps_clip: float = 0.2):
        ray.get([actor.set_loss_function.remote(eps_clip) for actor in self._actor_handlers])

    def save_checkpoint(self, save_path, should_save_optimizer):
        ray.get([actor.save_checkpoint.remote(save_path, should_save_optimizer) for actor in self._actor_handlers])


class PPOCriticRayActorGroup(TrainableModelRayActorGroup):

    def __init__(self, num_nodes, num_gpus_per_node) -> None:
        super().__init__(num_nodes, num_gpus_per_node, RayPPOCritic)

    def async_calculate_value(self, sequences_attention_mask_action_mask_refs):
        num_actors = len(self._actor_handlers)
        value_refs = []
        for i in range(len(sequences_attention_mask_action_mask_refs)):
            value_ref = self._actor_handlers[i % num_actors].calculate_value.remote(
                sequences_attention_mask_action_mask_refs[i])
            value_refs.append(value_ref)
        return value_refs

    def set_loss_function(self, value_clip: float = 0.4):
        ray.get([actor.set_loss_function.remote(value_clip) for actor in self._actor_handlers])


class PPOInitialRayActorGroup(PPORayActorGroup):

    def __init__(self, num_nodes, num_gpus_per_node) -> None:
        super().__init__(num_nodes, num_gpus_per_node, RayPPOInitialModel)

    def async_calculate_base_action_log_probs(self, sequences_attention_mask_action_mask_refs):
        num_actors = len(self._actor_handlers)
        base_action_log_probs_refs = []
        for i in range(len(sequences_attention_mask_action_mask_refs)):
            base_action_log_probs_ref = self._actor_handlers[i % num_actors].calculate_base_action_log_probs.remote(
                sequences_attention_mask_action_mask_refs[i])
            base_action_log_probs_refs.append(base_action_log_probs_ref)
        return base_action_log_probs_refs


class PPORewardRayActorGroup(PPORayActorGroup):

    def __init__(self, num_nodes, num_gpus_per_node) -> None:
        super().__init__(num_nodes, num_gpus_per_node, RayPPORewardModel)

    def async_calculate_r(self, sequences_attention_mask_action_mask_refs):
        num_actors = len(self._actor_handlers)
        r_refs = []
        for i in range(len(sequences_attention_mask_action_mask_refs)):
            r_ref = self._actor_handlers[i % num_actors].calculate_r.remote(
                sequences_attention_mask_action_mask_refs[i])
            r_refs.append(r_ref)
        return r_refs


def main(args):
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    if args.model == 'gpt2':
        actor_model_class, critic_model_class = GPTActor, GPTCritic
    elif args.model == 'bloom':
        actor_model_class, critic_model_class = BLOOMActor, BLOOMCritic
    elif args.model == 'opt':
        actor_model_class, critic_model_class = OPTActor, OPTCritic
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    logging.info("Start creating actors")
    # Initialize 4 models (actor, critic, initial_model and reward_model)
    actor_group = PPOActorRayActorGroup(num_nodes=args.num_actor_nodes, num_gpus_per_node=args.num_gpus_per_node)
    critic_group = PPOCriticRayActorGroup(num_nodes=args.num_critic_nodes, num_gpus_per_node=args.num_gpus_per_node)
    initial_group = PPOInitialRayActorGroup(num_nodes=args.num_initial_nodes, num_gpus_per_node=args.num_gpus_per_node)
    reward_group = PPORewardRayActorGroup(num_nodes=args.num_reward_nodes, num_gpus_per_node=args.num_gpus_per_node)
    logging.info("Actors created")

    # Prepare model for training
    generate_kwargs = {'max_length': 128, 'do_sample': True, 'temperature': 1.0, 'top_k': 50}
    ray.get(
        actor_group.async_init_model_from_pretrained(args.strategy, actor_model_class, args.pretrain, True) +
        critic_group.async_init_model_from_pretrained(args.strategy, critic_model_class, args.pretrain, True) +
        initial_group.async_init_model_from_pretrained(args.strategy, actor_model_class, args.pretrain, False) +
        reward_group.async_init_model_from_pretrained(args.strategy, critic_model_class, args.pretrain, False) +
        actor_group.async_prepare_for_sequence_generation(args.model, args.pretrain, generate_kwargs))
    logging.info("Models prepared for training")

    # Prepare models for training
    actor_group.load_csv_prompt_file_from_url_to_sampler(args.prompt_csv_url)
    actor_group.set_loss_function()
    critic_group.set_loss_function()
    # Training parameter
    num_episodes = args.num_episodes
    max_timesteps = args.max_timesteps
    update_timesteps = args.update_timesteps
    experience_batch_size = args.experience_batch_size
    # Start training
    logging.info("Training start")
    # Set all models to eval and add experience maker
    all_ray_actors = actor_group._actor_handlers + critic_group._actor_handlers + \
        initial_group._actor_handlers + reward_group._actor_handlers
    num_ray_actors = len(all_ray_actors)
    ray.get([ray_actor.eval.remote() for ray_actor in all_ray_actors])
    ray.get([ray_actor.add_experience_maker.remote() for ray_actor in all_ray_actors])
    # Used as a queue to coordinate experience making
    experience_composition_refs = []
    time = 0
    for episode in range(num_episodes):
        logging.info("episode {} started".format(episode))
        for _ in range(max_timesteps):
            time += 1
            # Experience queueing stage
            sequences_attention_mask_action_mask_refs = actor_group.async_sample_prompts_and_make_sequence(
                experience_batch_size)
            base_action_log_probs_refs = initial_group.async_calculate_base_action_log_probs(
                sequences_attention_mask_action_mask_refs)
            values_refs = critic_group.async_calculate_value(sequences_attention_mask_action_mask_refs)
            r_refs = reward_group.async_calculate_r(sequences_attention_mask_action_mask_refs)
            action_log_probs_refs = actor_group.async_calculate_action_log_probs(
                sequences_attention_mask_action_mask_refs)
            experience_composition_refs.extend([
                ExperienceCompositionRefs(sequences_attention_mask_action_mask_refs[i], action_log_probs_refs[i],
                                          base_action_log_probs_refs[i], values_refs[i], r_refs[i])
                for i in range(len(sequences_attention_mask_action_mask_refs))
            ])
            # Learning stage
            if time % update_timesteps == 0:
                experience_refs = []
                # calculate experiences
                for i, experience_composition_ref in enumerate(experience_composition_refs):
                    exp_composition_ref = experience_composition_ref
                    selected_ray_actor = all_ray_actors[i % num_ray_actors]
                    experience_refs.append(selected_ray_actor.make_experience.remote(exp_composition_ref))
                # backward
                ray.get(
                    actor_group.async_learn_on_experiences(experience_refs) +
                    critic_group.async_learn_on_experiences(experience_refs))
                # clear refs queue
                experience_composition_refs.clear()
    logging.info("Training finished")
    # Save checkpoint
    actor_group.save_checkpoint(args.save_path, args.need_optim_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_csv_url', type=str)
    parser.add_argument('--strategy',
                        choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='ddp')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt'])
    parser.add_argument('--pretrain', type=str, default='gpt2')
    parser.add_argument('--save_path', type=str, default='actor_checkpoint_prompts.pt')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--num_actor_nodes', type=int, help='num of nodes to use to host actor model', default=1)
    parser.add_argument('--num_critic_nodes', type=int, help='num of nodes to use to host critic model', default=1)
    parser.add_argument('--num_initial_nodes', type=int, help='num of nodes to use to host initial model', default=1)
    parser.add_argument('--num_reward_nodes', type=int, help='num of nodes to use to host reward model', default=1)
    parser.add_argument('--num_gpus_per_node', type=int, help='num of gpus on a ray node', default=1)
    args = parser.parse_args()
    ray.init()
    main(args)
