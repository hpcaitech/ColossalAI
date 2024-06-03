import torch
from torchvision.models import resnet50
from tqdm import tqdm

import colossalai
from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
from colossalai.device.device_mesh import DeviceMesh
from colossalai.legacy.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR


def synthesize_data():
    img = torch.rand(gpc.config.BATCH_SIZE, 3, 32, 32)
    label = torch.randint(low=0, high=10, size=(gpc.config.BATCH_SIZE,))
    return img, label


def main():
    colossalai.legacy.launch_from_torch(config="./config.py")

    logger = get_dist_logger()

    # trace the model with meta data
    model = resnet50(num_classes=10).cuda()

    input_sample = {"x": torch.rand([gpc.config.BATCH_SIZE * torch.distributed.get_world_size(), 3, 32, 32]).to("meta")}
    device_mesh = DeviceMesh(physical_mesh_id=torch.tensor([0, 1, 2, 3]), mesh_shape=[2, 2], init_process_group=True)
    model, solution = initialize_model(model, input_sample, device_mesh=device_mesh, return_solution=True)

    if gpc.get_global_rank() == 0:
        for node_strategy in solution:
            print(node_strategy)
    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    for epoch in range(gpc.config.NUM_EPOCHS):
        model.train()

        # if we use synthetic data
        # we assume it only has 10 steps per epoch
        num_steps = range(10)
        progress = tqdm(num_steps)

        for _ in progress:
            # generate fake data
            img, label = synthesize_data()

            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            train_loss = criterion(output, label)
            train_loss.backward(train_loss)
            torch.cuda.synchronize()
            optimizer.step()
        lr_scheduler.step()

        # run evaluation
        model.eval()
        correct = 0
        total = 0

        # if we use synthetic data
        # we assume it only has 10 steps for evaluation
        num_steps = range(10)
        progress = tqdm(num_steps)

        for _ in progress:
            # generate fake data
            img, label = synthesize_data()

            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                output = model(img)
                test_loss = criterion(output, label)
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}",
            ranks=[0],
        )


if __name__ == "__main__":
    main()
