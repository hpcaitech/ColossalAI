from pathlib import Path
from colossalai.logging import get_dist_logger
import colossalai
import torch
import os
from torch.fx import GraphModule
from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from tqdm import tqdm
from titans.utils import barrier_context
from colossalai.auto_parallel.tensor_shard.solver.cost_graph import CostGraph
from colossalai.auto_parallel.tensor_shard.solver.graph_analysis import GraphAnalyser
from colossalai.auto_parallel.tensor_shard.solver.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.solver.solver import Solver
from colossalai.auto_parallel.tensor_shard.solver.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer

DATA_ROOT = Path(os.environ.get('DATA', './data'))
BATCH_SIZE = 1024
NUM_EPOCHS = 10


def main():
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()

    with barrier_context():
        # build dataloaders
        train_dataset = CIFAR10(root=DATA_ROOT,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(size=32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                                ]))

    test_dataset = CIFAR10(root=DATA_ROOT,
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                           ]))

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        add_sampler=False,
        shuffle=True,
        batch_size=BATCH_SIZE,
        pin_memory=True,
    )

    test_dataloader = get_dataloader(
        dataset=test_dataset,
        add_sampler=False,
        batch_size=BATCH_SIZE,
        pin_memory=True,
    )

    # initialize device mesh
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # trace the model with meta data
    tracer = ColoTracer()
    model = resnet50(num_classes=10).cuda()
    input_sample = {'x': torch.rand([1024, 3, 32, 32]).to('meta')}
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    # prepare info for solver
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()
    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    graph_analyser = GraphAnalyser(gm)

    # solve the solution
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])
    if gpc.get_global_rank() == 0:
        for index, node in enumerate(graph.nodes):
            print(node.name, node.strategies_vector[solution[index]].name)

    # process the graph for distributed training ability
    gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict = runtime_preparation_pass(gm, solution, device_mesh)
    gm = runtime_apply_pass(gm)
    gm.recompile()

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        gm.train()
        if gpc.get_global_rank() == 0:
            train_dl = tqdm(train_dataloader)
        else:
            train_dl = train_dataloader
        for img, label in train_dl:
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = gm(img, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
            train_loss = criterion(output, label)
            train_loss.backward(train_loss)
            optimizer.step()
        lr_scheduler.step()

        gm.eval()
        correct = 0
        total = 0
        for img, label in test_dataloader:
            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                output = gm(img, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
                test_loss = criterion(output, label)
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}",
            ranks=[0])


if __name__ == '__main__':
    main()
