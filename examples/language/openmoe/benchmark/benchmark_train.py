import colossalai
import datasets
import torch
import transformers
from colossalai import get_default_parser
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import skip_init
from colossalai.utils import get_current_device
from model.modeling_openmoe import OpenMoeForCausalLM
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Adafactor
from transformers.models.llama import LlamaConfig
from utils import SimpleTimer, print_model_numel


class RandomDataset(Dataset):

    def __init__(self,
                 num_samples: int = 1000,
                 max_length: int = 2048,
                 vocab_size: int = 32000):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(0, vocab_size,
                                       (num_samples, max_length),
                                       device=get_current_device())
        self.attention_mask = torch.ones_like(self.input_ids,
                                              device=get_current_device())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx]
        }


def parse_args():
    parser = get_default_parser()
    # TODO: add model_name
    # parser.add_argument("--model_name", type=str, default="base", choices=["base", "8b"],
    #                     help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--num_epoch", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (per dp group) for the training dataloader.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples in the dataset.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    MDOEL_CONFIG = {
        "architectures": [
            "OpenMoeForCausalLM"
        ],
        "capacity_factor_eval": 2.0,
        "capacity_factor_train": 1.25,
        "drop_tks": True,
        "dropout_rate": 0.0,
        "expert_parallel": None,
        "gated": True,
        "head_dim": 64,
        "hidden_act": "swiglu",
        "hidden_size": 768,
        "intermediate_size": 2048,
        "label_smoothing": 0.0,
        "layer_norm_epsilon": 1e-06,
        "min_capacity": 4,
        "moe_layer_interval": 4,
        "noisy_policy": None,
        "num_attention_heads": 12,
        "num_experts": 16,
        "num_hidden_layers": 12,
        "num_key_value_heads": 12,
        "pretraining_tp": 1,
        "rope_scaling": None,
        "router_aux_loss_factor": 0.01,
        "router_z_loss_factor": 0.0001,
        "topk": 2,
        "torch_dtype": "float32",
        "vocab_size": 256384,
        "z_loss_factor": 0.0001
    }
    OPTIM_CONFIG = {
        "decay_rate": -0.8,
        "weight_decay": 0.01,
    }

    # update config from args
    for k in MDOEL_CONFIG:
        if hasattr(args, k):
            MDOEL_CONFIG[k] = getattr(args, k)

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()

    # Set up moe
    MOE_MANAGER.setup(seed=42, parallel="EP")

    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Build OpenMoe model
    config = LlamaConfig()
    for k, v in MDOEL_CONFIG.items():
        setattr(config, k, v)

    with skip_init():
        model = OpenMoeForCausalLM(config)

    logger.info(f"Finish init model with config:\n{config}", ranks=[0])
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model param count: {model_param/1e6:.2f}M", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Set plugin
    plugin = LowLevelZeroPlugin(initial_scale=2**5, stage=2)
    logger.info(f"Set plugin as {plugin}", ranks=[0])

    # Prepare tokenizer and dataloader
    dataset = RandomDataset(num_samples=args.num_samples)
    dataloader = plugin.prepare_dataloader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=True)

    # Set optimizer
    optimizer = Adafactor(model.parameters(),
                          decay_rate=OPTIM_CONFIG["decay_rate"],
                          weight_decay=OPTIM_CONFIG["weight_decay"])

    # Set booster
    booster = Booster(plugin=plugin)
    model, optimizer, _, dataloader, _ = booster.boost(model=model,
                                                       optimizer=optimizer,
                                                       dataloader=dataloader)

    # Start benchmark
    model.train()
    logger.info(f"Start benchmark", ranks=[0])

    timer = SimpleTimer()
    for epoch in range(args.num_epoch):
        for batch in tqdm(dataloader,
                          desc=f'Epoch [{epoch + 1}]',
                          disable=not coordinator.is_master()):
            timer.start("train_step")

            # Forward
            timer.start("forward")
            outputs = model(use_cache=False, chunk_head=True, **batch)
            loss = outputs['loss']
            torch.cuda.synchronize()
            timer.stop("forward")

            # Backward
            timer.start("backward")
            booster.backward(loss, optimizer)
            torch.cuda.synchronize()
            timer.stop("backward")

            # Optimizer step
            timer.start("optimizer_step")
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            timer.stop("optimizer_step")

            timer.stop("train_step")

    logger.info(f"Benchmark result:\n{repr(timer)}", ranks=[0])


if __name__ == "__main__":
    main()
