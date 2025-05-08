import torch
import torch.distributed as dist
from coati.dataset.loader import RawConversationDataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, Plugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
GRADIENT_ACCUMULATION_STEPS = 1
DATA_PATH = "/home/duanjunwen/datasets/math_dataset.jsonl"
MODEL_PATH = "/home/duanjunwen/models/Qwen/Qwen2.5-14B"
Device = torch.device("npu" if torch.npu.is_available() else "cpu")

class RandomDataset(Dataset):
    def __init__(self, num_samples, sequence_length, vocab_size=10000):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.input_idx = torch.randint(0, vocab_size, (num_samples, sequence_length))
        self.attention_mask = torch.randint(0, 2, (num_samples, sequence_length), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"input_ids": self.input_idx[idx], "attention_mask": self.attention_mask[idx]}

def load_model_and_tokenizer():
    attn_impl = "eager" if get_accelerator().name == "npu" else "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model = Qwen2ForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer, model

def all_reduce_mean(loss: torch.Tensor, plugin: Plugin) -> torch.Tensor:
    loss = loss.data
    group = getattr(plugin, "dp_group", None)
    dist.all_reduce(loss, group=group)
    return loss / dist.get_world_size(group)

def test_hybrid_qwen():
    colossalai.launch_from_torch()
    get_accelerator()
    coordinator = DistCoordinator()
    tokenizer, model = load_model_and_tokenizer()
    # dataset = RandomDataset(num_samples=100, sequence_length=2304)
    dataset = RawConversationDataset(tokenizer, DATA_PATH, 1024,  system_prompt="Please reason step by step, and put your final answer within \\boxed{}.")
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)
    plugin = HybridParallelPlugin(
        tp_size=8, 
        pp_size=1, 
        precision="bf16", 
        zero_stage=2, 
        cpu_offload=True,
    )
    # plugin = HybridParallelPlugin(tp_size=2, pp_size=2, precision="bf16", zero_stage=1, num_microbatches=4, enable_flash_attention=True)

    dataloader = plugin.prepare_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    booster = Booster(plugin=plugin)

    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, None, dataloader)

    def is_master():
        if isinstance(plugin, HybridParallelPlugin) and plugin.pp_size > 1:
            return coordinator.rank == coordinator.world_size - 1
        return coordinator.is_master()

    #####
    # train
    #####
    model.train()

    for epoch in range(NUM_EPOCHS):
        if booster.plugin.pp_size > 1:
            data_iter = iter(dataloader)
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Step",
                disable=not is_master(),
            )
            for step in step_bar:
                print(f"data_iter {data_iter}")
                outputs = booster.execute_pipeline(
                    data_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs[0],
                    optimizer=optimizer,
                    return_loss=True,
                )
                loss = outputs["loss"]
                if booster.plugin.stage_manager.is_last_stage():
                    global_loss = all_reduce_mean(loss, plugin)

                optimizer.step()

                if booster.plugin.stage_manager.is_last_stage():
                    grad_norm = optimizer.get_grad_norm()
                    step_bar.set_postfix({"loss": global_loss.item(), "grad_norm": grad_norm})

                optimizer.step()
                optimizer.zero_grad()
        else:
            total_loss = 0
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device=model.module.device)
                attention_mask = batch["attention_mask"].to(device=model.module.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                print(f"loss {loss}")
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                booster.backward(loss, optimizer)
                print(f"finish backward")
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"finish optimizer step")

                total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    test_hybrid_qwen()
