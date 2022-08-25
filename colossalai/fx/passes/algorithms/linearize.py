from torch.fx import GraphModule


def linearize(gm: GraphModule) -> dict:
    status_dict = {}
    node_dict = {}
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

        # first node in graph
        if last_dict_len == 0:
            node_dict[node_idx] = [node]
            status_dict[node.name] = list(node.users)

            continue

        # boundary case
        if len(status_dict) == 0:
            # current node region end point = next node region start point
            if last_dict_len == 1:
                node_idx += 1
                node_dict[node_idx] = [node]
                status_dict[node.name] = list(node.users)

                continue
            else:
                node_dict[node_idx].append(node)
                status_dict[node.name] = list(node.users)

                continue

        else:
            # in-node case
            node_dict[node_idx].append(node)
            status_dict[node.name] = list(node.users)

            continue

    return node_dict
