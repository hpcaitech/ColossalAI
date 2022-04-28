import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor


@colo_op_impl(torch.nn.functional.layer_norm)
def colo_layernorm(types, args=(), kwargs=None, pg=None):
    arg_num = len(args)
    if arg_num > 0:
        input_tensor = args[0]
    if arg_num > 1:
        normalized_shape = args[1]
    if arg_num > 2:
        weight = args[3]
    if arg_num > 3:
        bias = args[4]
    if arg_num > 4:
        eps = args[5]

    if 'input' in kwargs:
        input_tensor = kwargs['input']
    if 'weight' in kwargs:
        weight = kwargs['weight']
    if 'bias' in kwargs:
        bias = kwargs['bias']
    if 'eps' in kwargs:
        eps = kwargs['eps']

    if isinstance(input_tensor, ColoTensor):
        if input_tensor.is_activation() and not input_tensor.is_gathered():
            input_tensor.gather()
        input_tensor = input_tensor.torch_tensor()
    if isinstance(weight, ColoTensor):
        weight = weight.torch_tensor()
    if isinstance(bias, ColoTensor):
        bias = bias.torch_tensor()

    return ColoTensor.init_from_torch_tensor(
        torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, eps))
