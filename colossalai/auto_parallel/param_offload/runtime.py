import torch
from torch.fx.node import Node

from colossalai.gemini.tensor_utils import alloc_storage, free_storage

from colossalai.auto_parallel.param_offload.util import ModelParameters


class UploadParameter(torch.autograd.Function):
    """
    A customized upload operation which forward is parameter upload operation,
    backward is a parameter release operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices):
        # offload
        ctx.params_indices = params_indices
        for param_idx in params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            if fp16_param.data.device.type == "cpu":
                fp16_param.data = fp16_param.data.to("cuda")
            else:
                alloc_storage(fp16_param.data)
                fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # upload
        for param_idx in ctx.params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            free_storage(fp16_param.data)
        return grad_output, None


class OffloadParameter(torch.autograd.Function):
    """
    A customized offload operation which forward is parameter release operation,
    backward is a parameter upload operation.

    Args:
        input_: input matrix.
        params_indices:.
    """

    @staticmethod
    def forward(ctx, input_, params_indices):
        # offload
        ctx.params_indices = params_indices
        for param_idx in params_indices:
            free_storage(ModelParameters.fp16_params[param_idx].data)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # upload
        for param_idx in ctx.params_indices:
            fp16_param = ModelParameters.fp16_params[param_idx]
            alloc_storage(fp16_param.data)
            fp16_param.data.copy_(ModelParameters.fp32_master_params[param_idx].data)
        return grad_output, None

def convert_upload_to_action(tensor, params_indices):
    '''
    Convert UploadSpec into runtime action, implement upload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return UploadParameter.apply(tensor, params_indices)

def convert_offload_to_action(tensor, params_indices):
    '''
    Convert OffloadSpec into runtime action, implement offload operation target tensor.

    Argument:
        tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
    '''
    return OffloadParameter.apply(tensor, params_indices)


def replace_node_users(orig_node: Node, inserted_node: Node):
    user_list = list(orig_node.users.keys())
    for user in user_list:
        if user == inserted_node:
            continue
        new_args = list(user.args)
        new_kwargs = dict(user.kwargs)
        # the origin node may be a positional argument or key word argument of user node
        if orig_node in new_args:
            # substitute the origin node with offload_apply_node
            new_args[new_args.index(orig_node)] = inserted_node
            user.args = tuple(new_args)
        elif str(orig_node) in new_kwargs:
            # substitute the origin node with offload_apply_node
            new_kwargs[str(orig_node)] = inserted_node
            user.kwargs = new_kwargs


def runtime_offload_apply_pass(gm: torch.fx.GraphModule):
    """
    This pass is used to add the offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    for node in nodes:
        if node.node_info.has_param:
            param_indices = node.node_info.param_indices
            assert isinstance(param_indices, list)

            last_inp_node = list(node._input_nodes.keys())[-1]
            # with mod_graph.inserting_before(node) may not work
            with mod_graph.inserting_after(last_inp_node):
                upload_apply_node = mod_graph.create_node('call_function', convert_upload_to_action,
                                                          args=(last_inp_node, param_indices))
            replace_node_users(last_inp_node, upload_apply_node)

            if node.node_info.offload_param_flag:
                with mod_graph.inserting_after(node):
                    offload_apply_node = mod_graph.create_node('call_function', convert_offload_to_action,
                                                               args=(node, param_indices))
                replace_node_users(node, offload_apply_node)
    return gm



