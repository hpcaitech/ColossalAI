from typing import OrderedDict
from torch.fx import GraphModule
from collections import OrderedDict
import pdb


def linearize(gm: GraphModule) -> dict:
    status_dict = {}
    node_dict = OrderedDict()
    node_idx = 0
    for node in gm.graph.nodes:
        last_dict_len = len(status_dict)
        # remove node from users list in status_dict
        for item in status_dict.values():
            if node in item:
                item.remove(node)

        # pop node from status_dict if it is fully used
        for key in list(status_dict):
            if len(status_dict[key]) == 0:
                status_dict.pop(key)

        # first node in graph, it should be in n0-n1 type,
        # where n0 contains only input op, i.e. placeholder
        if last_dict_len == 0:
            node_dict[node_idx] = [node]
            status_dict[node.name] = list(node.users)
            node_idx += 1
            node_dict[node_idx] = []

            continue

        # boundary case
        if len(status_dict) == 0:
            # current node region end point = next node region start point
            # i.e. n1-n2-n3-... type node, each node contains only one op
            if last_dict_len == 1:
                if len(node_dict[node_idx]) > 0:
                    node_idx += 1
                    node_dict[node_idx] = []
                node_dict[node_idx].append(node)
                status_dict[node.name] = list(node.users)

                continue

            # n1-n2-n3, if n1 has multiple ops, the last op in n1 will be
            # the one who is able to clean all others in status_dict
            # and as the last_dict_len > 1, there are multiple ops are used
            # by this node, we view it as the end of one node and start a new node
            else:

                node_dict[node_idx].append(node)
                status_dict[node.name] = list(node.users)
                node_idx += 1
                node_dict[node_idx] = []

                continue

        else:
            # currently I will use bigger node structure
            # if the following region is activated, the node will be smaller
            #################################################
            # if last_dict_len == 1:
            #     if len(node_dict[node_idx]) > 0:
            #         node_idx += 1
            #     node_dict[node_idx] = [node]
            #     status_dict[node.name] = list(node.users)
            #
            #     continue
            #################################################

            # in-node case, as the current node can not clean status_dict
            # we view it as in-node status, the node will be appended to the
            # current node_idx
            node_dict[node_idx].append(node)
            status_dict[node.name] = list(node.users)

            continue

    # If the output node use multiple nodes, there might be an
    # empty node after the output node
    if len(node_dict[node_idx]) == 0:
        node_dict.pop[node_idx]
        node_idx -= 1

    # pop the last two nodes
    node_dict.pop(0)
    node_dict.pop(node_idx)
    return node_dict
