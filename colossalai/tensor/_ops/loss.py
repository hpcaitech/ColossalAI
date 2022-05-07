from colossalai.tensor.spec import ShardPattern
import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor
from colossalai.nn.loss.loss_1d import VocabParallelCrossEntropyLoss1D

@colo_op_impl(torch.nn.functional.cross_entropy)
def colo_cross_entropy(types, args=(), kwargs=None, pg=None):
    arg_num = len(args)

    if arg_num > 0:
        input_tensor = args[0]
    if arg_num > 1:
        target = args[1]
    if arg_num > 2:
        weight = args[2]

    if 'input' in kwargs:
        input_tensor = kwargs.pop('input')
    if 'target' in kwargs:
        target = kwargs.pop('target')
    if 'weight' in kwargs:
        weight = kwargs.pop('weight')

    if not isinstance(input_tensor, ColoTensor):
        input_tensor = ColoTensor.init_from_torch_tensor(input_tensor)
    if isinstance(target, ColoTensor):
        target = target.torch_tensor()

    if input_tensor.is_gathered(): # Input is gathered
        # TODO(jzy) Shall we make the result of loss function a ColoTensor?
        return ColoTensor.init_from_torch_tensor(torch.nn.functional.cross_entropy(
            input_tensor.torch_tensor(), target, weight))
    elif input_tensor.has_spec() and input_tensor.shard_spec.num_action == 1:    # Single Model Parallel Applied
        if input_tensor.shard_pattern == ShardPattern.Col:
            return ColoTensor.init_from_torch_tensor(
                VocabParallelCrossEntropyLoss1D()(input_tensor.torch_tensor(), target))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
