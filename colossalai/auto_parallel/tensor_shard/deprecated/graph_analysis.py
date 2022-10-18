from dataclasses import dataclass
from typing import List

from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from colossalai.fx.passes.utils import get_node_module

__all__ = ['LiveVariable', 'LiveVariableVector', 'LiveStage', 'GraphAnalyser']


@dataclass
class LiveVariable:
    """
    LiveVariable is a data structure to store the meta information of a variable for liveness analysis.
    """
    name: str
    node: Node
    is_inplace: bool


class LiveVariableVector(list):
    """
    LiveVariableVector is a data structure to store the list of LiveVariable objects.
    """

    def exists(self, name) -> bool:
        """
        Check if a variable has already existed in the current list by name.
        """
        for var in self:
            if name == var.name:
                return True
        return False

    def get(self, name) -> LiveVariable:
        for var in self:
            if name == var.name:
                return var
        raise KeyError(f"Variable {name} is not found")

    def copy(self) -> "LiveVariableVector":
        """
        Create a copy of this vector
        """
        vector = LiveVariableVector()
        for var in self:
            vector.append(var)
        return vector


@dataclass
class LiveStage:
    """
    LiveStage is a data structure to record the living variables at this current node.
    """
    name: str
    node: Node
    all_live_vars: LiveVariableVector
    unique_live_vars: LiveVariableVector


class GraphAnalyser:

    def __init__(self, gm: GraphModule):
        self._gm = gm
        self._graph = gm.graph

    @property
    def gm(self) -> GraphModule:
        """
        Return the GraphModule object associated with this analyser.
        """
        return self._gm

    @property
    def graph(self) -> Graph:
        """
        Return the Graph object associated with this analyser.
        """
        return self._graph

    def liveness_analysis(self) -> List[LiveStage]:
        """
        Analyse the graph to obtain the variable liveness information. This function returns
        an ordered dictionary where the key is the compute stage ID and the value is a LivenessStage object.
        """
        compute_nodes = self.graph.nodes
        liveness_list = []

        # checked: record all variables created since the first stage
        # all: record the live variables only exist until the current stage.
        #       this can be different from the `checked list`` as some varialbes may be destroyed prior to this stage.
        # unique: record the unique live variables only exist until the current stage.
        #       this is different from `all list` as some variables are duplicated.
        checked_variables = LiveVariableVector()
        all_live_variables = LiveVariableVector()
        unique_live_vars = LiveVariableVector()

        for idx, node in enumerate(compute_nodes):
            #############################
            # find new living variables #
            #############################
            # detect whether the current op is an in-place op
            # if it is an in-place op, we would deem it as a duplciate var
            is_inplace = False
            if node.op == 'call_function':
                # check if this is an inplace op such as torch.nn.functional.relu(x, inplace=True)
                if node.kwargs.get('inplace', False):
                    is_inplace = True
            elif node.op == 'call_module':
                # to check if this is an inplace op such as torch.nn.Relu(inplace=True)
                module = get_node_module(node)
                if getattr(module, 'inplace', False):
                    is_inplace = True

            # add the output var
            meta = getattr(node, '_meta_data', None)
            live_var = LiveVariable(name=node.name, node=node, is_inplace=is_inplace)
            if not is_inplace:
                unique_live_vars.append(live_var)
            checked_variables.append(live_var)
            all_live_variables.append(live_var)

            # check if any input is not checked yet
            for arg in node.args:
                if not isinstance(arg, Node):
                    continue
                arg_name = arg.name
                if not checked_variables.exists(arg_name):
                    live_var_from_arg = LiveVariable(name=arg_name, node=node, is_inplace=False)
                    all_live_variables.append(live_var_from_arg)
                    checked_variables.append(live_var_from_arg)
                    unique_live_vars.append(live_var_from_arg)

            # TODO: add the logic to remove live variables
            # this should be completed if we are able to trace the backward compute graph

            # add this stage to liveness dict
            stage = LiveStage(name=node.name,
                              node=node,
                              all_live_vars=all_live_variables.copy(),
                              unique_live_vars=unique_live_vars.copy())
            # if a LiveStage is covered by another LiveStage, we just keep the larger one.
            replace = False
            for index, prev_stage in enumerate(liveness_list):
                all_covered = True
                for ele in prev_stage.unique_live_vars:
                    if ele not in stage.unique_live_vars:
                        all_covered = False
                        break
                if all_covered:
                    replace = True
                    break
            if replace:
                liveness_list[index] = stage
            else:
                liveness_list.append(stage)

        return liveness_list

    def get_alias_set(self):
        pass
