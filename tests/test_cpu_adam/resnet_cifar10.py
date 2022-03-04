
import torch
import torch.distributed as dist
import colossalai
import argparse
from tqdm import tqdm
from torchvision.models import resnet50
from colossalai.context.parallel_mode import ParallelMode
from colossalai.zero import ShardedOptimizer
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from colossalai.nn.optimizer import CPUAdam
from colossalai.utils import Timer



BATCH_SIZE = 64
NUM_EPOCHS = 100

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def sync_param(model):
    for param in model.parameters():
        dist.broadcast(param, src=0, group=gpc.get_group(ParallelMode.DATA))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log')
    parser.add_argument('--use_torch', action='store_true')
    parser.add_argument('--local_rank')
    return parser.parse_args()


def check_model_is_synced(model, optim):
    rank = gpc.get_local_rank(ParallelMode.DATA)
    world_size = gpc.get_world_size(ParallelMode.DATA)

    if isinstance(optim, ShardedOptimizer):
        for param_group in optim._optimizer.param_groups:
            for param in param_group['params']:
                param_tmp = param.clone().detach()
                tensor_list = [torch.empty_like(param_tmp) for _ in range(world_size - 1)]
                tensor_list.insert(rank, param_tmp)

                dist.all_gather(tensor_list, param_tmp, group=gpc.get_group(ParallelMode.DATA))

                for i in range(world_size - 1):
                    assert torch.all(tensor_list[i] == tensor_list[i + 1])
    else:
        for param in model.parameters():
            param_tmp = param.clone().detach()
            tensor_list = [torch.empty_like(param_tmp) for _ in range(world_size - 1)]
            tensor_list.insert(rank, param_tmp)

            dist.all_gather(tensor_list, param_tmp, group=gpc.get_group(ParallelMode.DATA))

            for i in range(world_size - 1):
                assert torch.all(tensor_list[i] == tensor_list[i + 1])


def count_num_param_with_grad(optim):
    num = 0
    for group_id in range(len(optim._fp16_param_groups)):
        param_group = optim._fp16_param_groups[group_id]

        for param in param_group:
            if param.grad is not None:
                num += 1
    return num

timer = Timer()

def main():
    disable_existing_loggers()
    args = parse_args()

    # init dist
    colossalai.launch_from_torch(config=dict())

    # create logger
    logger = get_dist_logger()
    logger.info("dist initialized", ranks=[0])

    if gpc.get_global_rank() == 0 and args.log:
        logger.log_to_file(args.log, mode='w')

    # create resnet
    model = resnet50(num_classes=10).cuda()

    if args.use_torch:
        model = DDP(model)
    else:
        model = model.half()
        sync_param(model)


    # create optim
    # optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = CPUAdam(model.parameters(), lr=1e-3)
    if not args.use_torch:
        optim = ShardedOptimizer(optim,
                                 clip_grad_norm=0,
                                 verbose=True,
                                 initial_scale=2**10,
                                 overlap_communication=False,
                                 partition_grad=False,
                                 cpu_offload=True,
                                 cpu_fp16_param=True,
                                 cpu_fp16_grad=True,
                                 )
    logger.info("optim created", ranks=[0])

    # create dataloader
    train_dataset = CIFAR10(root='~/data',
                            download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(size=32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                            ]))

    test_dataset = CIFAR10(root='~/data',
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                           ]))

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        add_sampler=True,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
    )

    test_dataloader = get_dataloader(
        dataset=test_dataset,
        add_sampler=False,
        batch_size=BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
    )

    logger.info(f"num of parameters: {len(list(model.parameters()))}", ranks=[0])

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        timer.start()

        model.train()
        acc_train_loss = torch.Tensor([0.0]).cuda()
        for img, label in train_dataloader:
            img = img.cuda()
            label = label.cuda()

            if not args.use_torch:
                img = img.half()

            optim.zero_grad()
            output = model(img)

            if not args.use_torch:
                output = output.float()

            train_loss = criterion(output, label)
            acc_train_loss.add_(train_loss)

            if isinstance(optim, ShardedOptimizer):
                optim.backward(train_loss) 
                optim.sync_grad()
            else:
                train_loss.backward()
            optim.step()

        epoch_time = timer.stop()

        model.eval()
        correct = 0
        total = 0
        acc_test_loss = torch.Tensor([0.0]).cuda()
        for img, label in test_dataloader:
            img = img.cuda()
            label = label.cuda()

            if not args.use_torch:
                img = img.half()

            with torch.no_grad():
                output = model(img)
                
                if not args.use_torch:
                    output = output.float()

                test_loss = criterion(output, label)

            acc_test_loss.add_(test_loss)
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

        logger.info(
            f"Epoch {epoch} - train loss: {acc_train_loss / len(train_dataloader)}, "
            f"test loss: {acc_test_loss / len(test_dataloader)}, total: {total}, "
            f"correct: {correct}, epoch time: {epoch_time}"  )


if __name__ == '__main__':
    main()
