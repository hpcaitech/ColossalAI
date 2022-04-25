import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor


@colo_op_impl(torch.nn.functional.cross_entropy)
def colo_cross_entropy(types, args=(), kwargs=None, pg=None):
    arg_num = len(args)

    if arg_num > 0:
        input_tensor = args[0]
    if arg_num > 1:
        target = args[1]
    if arg_num > 2:
        weight = args[3]

    if 'input' in kwargs:
        input_tensor = kwargs['input']
    if 'target' in kwargs:
        target = kwargs['target']
    if 'weight' in kwargs:
        weight = kwargs['weight']

    if isinstance(input_tensor, ColoTensor):
        input_tensor = input_tensor.torch_tensor()
    if isinstance(target, ColoTensor):
        target = target.torch_tensor()

    return ColoTensor.init_from_torch_tensor(torch.nn.functional.cross_entropy(input_tensor, target, weight))
