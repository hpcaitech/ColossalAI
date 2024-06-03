from typing import List

import torch
from torch.fx.node import Node

from .region import Region
from .util import GlobalRuntimeInfo, requires_upload_p_in_fwd


class SynPreFwdPostBwdOP(torch.autograd.Function):
    """
    A customized prefetch and offload operation.

    Args:
        input_: input tensor.
        fwd_info: information dict, which contains region indices
            that need to be uploaded or freed during forward pass.
        bwd_info: information dict, which contains region indices
            that need to be uploaded during backward pass.
    """

    @staticmethod
    def forward(ctx, input_, fwd_info, bwd_info):
        ctx.bwd_info = bwd_info
        d2h_rid = fwd_info.get("d2h_rid", None)
        if d2h_rid is not None:
            free_region = GlobalRuntimeInfo().region_list[d2h_rid]
            assert isinstance(free_region, Region)
            free_region.free_cuda_data()

        h2d_rid = fwd_info.get("h2d_rid", None)
        if h2d_rid is not None:
            h2d_region = GlobalRuntimeInfo().region_list[h2d_rid]
            assert isinstance(h2d_region, Region)
            h2d_region.move_param_to_cuda()

        return input_

    @staticmethod
    def backward(ctx, grad_output):
        h2d_rid = ctx.bwd_info.get("h2d_rid", None)
        if h2d_rid is not None:
            pref_region = GlobalRuntimeInfo().region_list[h2d_rid]
            assert isinstance(pref_region, Region)
            pref_region.move_param_to_cuda()

        return grad_output, None, None


class AsynPreFwdPostBwdOP(torch.autograd.Function):
    """
    A customized prefetch and offload operation.

    Args:
        input_: input tensor.
        fwd_info: information dict, which contains region indices
            that need to be prefetched, waited, or freed during forward pass.
        bwd_info: information dict, which contains region indices
            that need to be prefetched or waited during backward pass.
    """

    @staticmethod
    def forward(ctx, input_, fwd_info, bwd_info):
        ctx.bwd_info = bwd_info

        sync_rid = fwd_info.get("sync_rid", None)
        if sync_rid is not None:
            prefetch_event = GlobalRuntimeInfo().fwd_prefetch_event_map.get(sync_rid, None)
            if prefetch_event:
                prefetch_event.wait()

        h2d_rid = fwd_info.get("h2d_rid", None)
        if h2d_rid is not None:
            pref_region = GlobalRuntimeInfo().region_list[h2d_rid]
            assert isinstance(pref_region, Region)
            master_stream = torch.cuda.current_stream()
            with torch.cuda.stream(GlobalRuntimeInfo().h2d_stream):
                GlobalRuntimeInfo().h2d_stream.wait_stream(master_stream)
                pref_region.move_param_to_cuda()

            prefetch_event = torch.cuda.Event()
            prefetch_event.record(GlobalRuntimeInfo().h2d_stream)
            GlobalRuntimeInfo().fwd_prefetch_event_map[h2d_rid] = prefetch_event

        return input_

    @staticmethod
    def backward(ctx, grad_output):
        sync_rid = ctx.bwd_info.get("sync_rid", None)
        if sync_rid is not None:
            wait_region = GlobalRuntimeInfo().region_list[sync_rid]
            assert isinstance(wait_region, Region)
            prefetch_event = GlobalRuntimeInfo().bwd_prefetch_event_map.get(sync_rid, None)
            if prefetch_event:
                prefetch_event.wait()
            else:
                wait_region.move_param_to_cuda()

        h2d_rid = ctx.bwd_info.get("h2d_rid", None)
        if h2d_rid is not None:
            pref_region = GlobalRuntimeInfo().region_list[h2d_rid]
            assert isinstance(pref_region, Region)
            master_stream = torch.cuda.current_stream()
            with torch.cuda.stream(GlobalRuntimeInfo().h2d_stream):
                GlobalRuntimeInfo().h2d_stream.wait_stream(master_stream)
                pref_region.move_param_to_cuda()

            prefetch_event = torch.cuda.Event()
            prefetch_event.record(GlobalRuntimeInfo().h2d_stream)
            GlobalRuntimeInfo().bwd_prefetch_event_map[h2d_rid] = prefetch_event
        return grad_output, None, None


def convert_fwd_upload_bwd_offload_to_action(tensor, fwd_info, bwd_info):
    """
    Convert Upload and Offload operation into runtime action.

    Argument:
        tensor(torch.Tensor): input tensor.
        fwd_info(dict): information dict, which contains region indices
            that need to be uploaded, or freed during forward pass.
        bwd_info(dict): information dict, which contains region indices
            that need to be uploaded during backward pass.
    """
    with torch._C.DisableTorchFunction():
        ret = SynPreFwdPostBwdOP.apply(tensor, fwd_info, bwd_info)
    return ret


def convert_fwd_prefetch_bwd_offload_to_action(tensor, fwd_info, bwd_info):
    """
    Convert Prefetch and Offload operation into runtime action.

    Argument:
        tensor(torch.Tensor): input tensor.
        fwd_info(dict): information dict, which contains region indices
            that need to be prefetched, waited, or freed during forward pass.
        bwd_info(dict): information dict, which contains region indices
            that need to be prefetched or waited during backward pass.
    """
    with torch._C.DisableTorchFunction():
        ret = AsynPreFwdPostBwdOP.apply(tensor, fwd_info, bwd_info)
    return ret


def replace_node_users(orig_node: Node, inserted_node: Node, rep_user_nodes: List[Node] = None):
    user_list = list(orig_node.users.keys())
    if rep_user_nodes is not None:
        user_list = rep_user_nodes
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


def runtime_syn_offload_apply_pass(gm: torch.fx.GraphModule, region_list: List[Region]):
    """
    This pass is used to add the synchronous upload and offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    last_inp_node = tuple(mod_graph.nodes)[0]

    for r_idx, region in enumerate(region_list):
        # forward upload
        fwd_info = {}
        if requires_upload_p_in_fwd(region_list[region.shared_rid]):
            fwd_info["h2d_rid"] = region.r_id

        # forward offload
        if r_idx > 0 and region_list[r_idx - 1].need_offload:
            fwd_info["d2h_rid"] = r_idx - 1

        bwd_info = {}
        # backward upload
        if r_idx > 0 and region_list[r_idx - 1].need_offload:
            bwd_info["h2d_rid"] = region_list[r_idx - 1].r_id

        if fwd_info or bwd_info:
            with mod_graph.inserting_after(last_inp_node):
                new_node = mod_graph.create_node(
                    "call_function", convert_fwd_upload_bwd_offload_to_action, args=(last_inp_node, fwd_info, bwd_info)
                )
            replace_node_users(last_inp_node, new_node)

        last_inp_node = region.nodes[-1]

    return gm


def runtime_asyn_offload_apply_pass(gm: torch.fx.GraphModule, region_list: List[Region]):
    """
    This pass is used to add the asynchronous prefetch and offload spec apply node to the origin graph.
    """
    mod_graph = gm.graph

    # upload parameters of the first region
    last_inp_node = tuple(mod_graph.nodes)[0]
    first_region_with_p = [region for region in region_list if region.param_size][0]
    fwd_info = {"h2d_rid": first_region_with_p.r_id}
    with mod_graph.inserting_after(last_inp_node):
        upload_apply_node = mod_graph.create_node(
            "call_function", convert_fwd_upload_bwd_offload_to_action, args=(last_inp_node, fwd_info, {})
        )
    replace_node_users(last_inp_node, upload_apply_node)
    last_inp_node = upload_apply_node

    for r_idx, region in enumerate(region_list):
        # forward prefetch
        fwd_info = {}
        if region.param_size:
            fwd_info["sync_rid"] = region.r_id
        fwd_prefetch_region = region.fwd_prefetch_region
        if fwd_prefetch_region and requires_upload_p_in_fwd(region_list[fwd_prefetch_region.shared_rid]):
            fwd_info["h2d_rid"] = fwd_prefetch_region.r_id

        # forward offload
        if r_idx > 0 and region_list[r_idx - 1].need_offload:
            fwd_info["d2h_rid"] = r_idx - 1

        bwd_info = {}
        # backward prefetch
        if r_idx > 0 and region_list[r_idx - 1].need_offload:
            bwd_info["sync_rid"] = r_idx - 1
        if r_idx > 0 and region_list[r_idx - 1].bwd_prefetch_region:
            bwd_info["h2d_rid"] = region_list[r_idx - 1].bwd_prefetch_region.r_id

        if fwd_info or bwd_info:
            with mod_graph.inserting_after(last_inp_node):
                new_node = mod_graph.create_node(
                    "call_function",
                    convert_fwd_prefetch_bwd_offload_to_action,
                    args=(last_inp_node, fwd_info, bwd_info),
                )
            replace_node_users(last_inp_node, new_node)

        last_inp_node = region.nodes[-1]

    if region.bwd_prefetch_region:
        bwd_info = {"h2d_rid": region.bwd_prefetch_region.r_id}
        with mod_graph.inserting_after(last_inp_node):
            new_node = mod_graph.create_node(
                "call_function", convert_fwd_prefetch_bwd_offload_to_action, args=(last_inp_node, {}, bwd_info)
            )
        replace_node_users(last_inp_node, new_node)
    # gm.graph.print_tabular()
    return gm
