import torch
try:
    import torchvision.models as tm
except:
    pass
from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import split_with_split_nodes_pass, balanced_split_pass
from torch.fx import GraphModule

import random
import numpy as np
import inspect

MANUAL_SEED = 0
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True


@pytest.mark.skip('skip as torchvision is required')
def test_torchvision_models():
    MODEL_LIST = [
        tm.vgg11, tm.resnet18, tm.densenet121, tm.mobilenet_v3_small, tm.resnext50_32x4d, tm.wide_resnet50_2,
        tm.regnet_x_16gf, tm.vit_b_16, tm.convnext_small, tm.efficientnet_b0, tm.mnasnet0_5
    ]

    tracer = ColoTracer()
    data = torch.rand(2, 3, 224, 224)

    for model_cls in MODEL_LIST:
        model = model_cls()
        model.eval()
        cpu_rng_state = torch.get_rng_state()
        output = model(data)
        graph = tracer.trace(root=model)
        gm = GraphModule(model, graph, model.__class__.__name__)
        gm.recompile()

        # apply transform passes
        annotated_model = balanced_split_pass(gm, 2)
        split_model, split_submodules = split_with_split_nodes_pass(annotated_model)

        # get split model
        model_part0 = list(split_model.children())[0]
        model_part1 = list(split_model.children())[1]

        # set rng state and compute output of split model
        torch.set_rng_state(cpu_rng_state)
        output_part0 = model_part0(data)
        sig = inspect.signature(model_part1.forward)
        if isinstance(output_part0, torch.Tensor):
            output_part1 = model_part1(output_part0)
        else:
            if len(output_part0) > len(sig.parameters):
                output_part0 = output_part0[:len(sig.parameters)]
            output_part1 = model_part1(*output_part0)
        assert output.equal(output_part1)


if __name__ == '__main__':
    test_torchvision_models()
