:warning: **This content may be outdated since the major update of Colossal Chat. We will update this content soon.**

# Distributed PPO Training on Stage 3

## Detach Experience Makers and Trainers

We can completely separate the trainers and makers.

<p align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/chat/basic_structure.png?raw=true" width=600/>
</p>

- The experience maker performs inference, produces experience, and remotely delivers it to the trainer (1).
- The trainer consumes experience to train models, and periodically transmits new model parameters to the maker (2.1, 2.2).
- Using an experience buffer to overlap transmission and computing.

In this manner, each node will work continuously without model idle time, and different optimization strategies can be applied for inference and training to meet the needs of speed or storage. It is also helpful for scalability.

`DetachedPPOTrainer` and `ExperienceMakerHolder` are Ray Actors (distinguished from Actor Model), representing Trainer and Experience Maker on the graph above, respectively.

[More about Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html)

## Usage

See examples at `ColossalAI/application/Chat/examples/ray`

### Setup Makers

- define makers' environment variables :

  ```python
  env_info_makers = [{
      'local_rank': '0',
      'rank': str(rank),
      'world_size': str(num_makers),
      'master_port': maker_port,
      'master_addr': master_addr
  } for rank in range(num_makers)]

  ```

- define maker models :

  ```python
  def model_fn():
      actor = get_actor_from_args(...)
      critic = get_critic_from_args(...)
      reward_model = get_reward_model_from_args(...)
      initial_model = get_actor_from_args(...)
      return actor, critic, reward_model, initial_model

  ```

- set experience_holder_refs :

  ```python
  experience_holder_refs = [
      ExperienceMakerHolder.options(
          name=f"maker_{i}",
          num_gpus=1,
          max_concurrency=2
      ).remote(
          detached_trainer_name_list=[f"trainer_{x}" for x in target_trainers(...)],
          model_fn=model_fn,
          ...)
      for i, env_info_maker in enumerate(env_info_makers)
  ]
  ```

  The names in the `detached_trainer_name_list` refer to the target trainers that the maker should send experience to.
  We set a trainer's name the same as a maker, by `.options(name="str")`. See below.

### Setup Trainers

- define trainers' environment variables :
  ```python
  env_info_trainers = [{
      'local_rank': '0',
      'rank': str(rank),
      'world_size': str(num_trainers),
      'master_port': trainer_port,
      'master_addr': master_addr
  } for rank in range(num_trainers)]
  ```
- define trainer models :

  ```python
  def trainer_model_fn():
      actor = get_actor_from_args(...)
      critic = get_critic_from_args(...)
      return actor, critic
  ```

- set trainer_refs :
  ```python
  trainer_refs = [
      DetachedPPOTrainer.options(
          name=f"trainer{i}",
          num_gpus=1,
          max_concurrency=2
      ).remote(
          experience_maker_holder_name_list=[f"maker{x}" for x in target_makers(...)],
          model_fn = trainer_model_fn(),
          ...)
      for i, env_info_trainer in enumerate(env_info_trainers)
  ]
  ```
  The names in `experience_maker_holder_name_list` refer to the target makers that the trainer should send updated models to.
  By setting `detached_trainer_name_list` and `experience_maker_holder_name_list`, we can customize the transmission graph.

### Launch Jobs

- define data_loader :

  ```python
  def data_loader_fn():
      return = torch.utils.data.DataLoader(dataset=dataset)

  ```

- launch makers :

  ```python
  wait_tasks = []
  for experience_holder_ref in experience_holder_refs:
      wait_tasks.append(
          experience_holder_ref.workingloop.remote(data_loader_fn(),
                                                   num_steps=experience_steps))

  ```

- launch trainers :

  ```python
  for trainer_ref in trainer_refs:
      wait_tasks.append(trainer_ref.fit.remote(total_steps, update_steps, train_epochs))
  ```

- wait for done :
  ```python
  ray.get(wait_tasks)
  ```

## Flexible Structure

We can deploy different strategies to makers and trainers. Here are some notions.

### 2 Makers 1 Trainer

<p align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/chat/2m1t.png?raw=true" width=600/>
</p>

### 2 Makers 2 Trainer

<p align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/chat/2m2t.png?raw=true" width=600/>
</p>

### Maker Inference Quantization

<p align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/chat/2m2t_quantize.png?raw=true" width=600/>
</p>

### Tensor Parallel

<p align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/chat/tp_ddp_hybrid.png?raw=true" width=600/>
</p>

## TODO

- [ ] Support LoRA
- [ ] Support TP & PP
