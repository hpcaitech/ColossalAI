import torch
import torch.distributed as dist
import transformers
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils import get_current_device
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator

from args import parse_demo_args
from data import BeansDataset, beans_collator


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator):
        
    torch.cuda.synchronize()
    model.train()

    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        
        for batch in pbar:

            # Foward
            optimizer.zero_grad()
            batch = move_to_cuda(batch, torch.cuda.current_device())
            outputs = model(**batch)
            loss = outputs['loss']

            # Backward
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()

            # Print batch loss
            pbar.set_postfix({'loss': loss.item()})


@torch.no_grad()
def evaluate_model(epoch, model, eval_dataloader, num_labels, coordinator):
    
    model.eval()
    accum_loss = torch.zeros(1, device=get_current_device())
    total_num = torch.zeros(1, device=get_current_device())
    accum_correct = torch.zeros(1, device=get_current_device())

    for batch in eval_dataloader:
        batch = move_to_cuda(batch, torch.cuda.current_device())
        outputs = model(**batch)
        val_loss, logits = outputs[:2]
        accum_loss += (val_loss / len(eval_dataloader))
        if num_labels > 1:
            preds = torch.argmax(logits, dim=1)
        elif num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        total_num += batch["labels"].shape[0]
        accum_correct += (torch.sum(preds == labels))

    dist.all_reduce(accum_loss)
    dist.all_reduce(total_num)
    dist.all_reduce(accum_correct)
    avg_loss = "{:.4f}".format(accum_loss.item())
    accuracy = "{:.4f}".format(accum_correct.item() / total_num.item())
    if coordinator.is_master():
        print(f"Evaluation result for epoch {epoch + 1}: \
                average_loss={avg_loss}, \
                accuracy={accuracy}.")
        
    
   

def main():

    args = parse_demo_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size

    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Prepare Dataset
    image_processor = ViTImageProcessor.from_pretrained(args.model_name_or_path)
    train_dataset = BeansDataset(image_processor, split='train')
    eval_dataset = BeansDataset(image_processor, split='validation')


    # Load pretrained ViT model
    config = ViTConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = train_dataset.num_labels
    config.id2label = {str(i): c for i, c in enumerate(train_dataset.label_names)}
    config.label2id = {c: str(i) for i, c in enumerate(train_dataset.label_names)}
    model = ViTForImageClassification.from_pretrained(args.model_name_or_path, 
                                                      config=config, 
                                                      ignore_mismatched_sizes=True)
    logger.info(f"Finish loading model from {args.model_name_or_path}", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Set plugin
    booster_kwargs = {}
    if args.plugin == 'torch_ddp_fp16':
        booster_kwargs['mixed_precision'] = 'fp16'
    if args.plugin.startswith('torch_ddp'):
        plugin = TorchDDPPlugin()
    elif args.plugin == 'gemini':
        plugin = GeminiPlugin(device=get_current_device(),
                        placement_policy='cpu',
                        pin_memory=True,
                        strict_ddp_mode=True,
                        initial_scale=2**5)
    elif args.plugin == 'low_level_zero':
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    logger.info(f"Set plugin as {args.plugin}", ranks=[0])

    # Prepare dataloader
    train_dataloader = plugin.prepare_dataloader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=beans_collator)
    eval_dataloader = plugin.prepare_dataloader(eval_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=beans_collator)

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=(args.learning_rate * world_size), weight_decay=args.weight_decay)

    # Set lr scheduler
    total_steps = len(train_dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=(len(train_dataloader) * args.num_epoch),
                                           warmup_steps=num_warmup_steps)

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, train_dataloader, lr_scheduler = booster.boost(model=model, 
                                                                  optimizer=optimizer, 
                                                                  dataloader=train_dataloader, 
                                                                  lr_scheduler=lr_scheduler)
    
    # Finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(args.num_epoch):
        train_epoch(epoch, model, optimizer, lr_scheduler, train_dataloader, booster, coordinator)
        evaluate_model(epoch, model, eval_dataloader, eval_dataset.num_labels, coordinator)
    logger.info(f"Finish finetuning", ranks=[0])

    # Save the finetuned model
    booster.save_model(model, args.output_path)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])


if __name__ == "__main__":
    main()