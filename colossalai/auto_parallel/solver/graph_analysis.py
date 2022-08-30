from dataclasses import dataclass
from torch.fx.node import Node
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from collections import OrderedDict as ODict
from typing import List, OrderedDict, Union, Any
from colossalai.fx.passes.utils import get_node_module

__all__ = ['LiveVariable', 'LiveVariableVector', 'LiveStage', 'GraphAnalyser']


@dataclass
class LiveVariable:
    """
    LiveVariable is a data structure to store the meta information of a variable for liveness analysis.
    """
    name: str
    meta: Union[Any, List[Any]]
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

    def liveness_analysis(self) -> OrderedDict[int, LiveStage]:
        """
        Analyse the graph to obtain the variable liveness information. This function returns
        an ordered dictionary where the key is the compute stage ID and the value is a LivenessStage object.
        """
        compute_nodes = self.graph.nodes
        liveness_dict = ODict()

        # checked: record all variables created since the first stage
        # all: record the live variables only exist until the current stage.
        #       this can be different from the `checked list`` as some varialbes may be destroyed prior to this stage.
        # unique: record the unique live variables only exist until the current stage.
        #       this is different from `all list` as some variables are duplicated.
        checked_variables = LiveVariableVector()
        all_live_variables = LiveVariableVector()
        unique_live_vars = LiveVariableVector()

        def _add_param_or_buf(node, tensor_type):
            module = get_node_module(node)

            if tensor_type == 'param':
                iterator = module.named_parameters()
            elif tensor_type == 'buffer':
                iterator = module.named_buffers()
            else:
                raise ValueError(f"Expected tensor_type to be param or buffer, but got {tensor_type}")

            for name, tensor in iterator:
                tensor_name = f'{node.name}.{name}'

                if not checked_variables.exists(tensor_name):
                    live_tensor = LiveVariable(name=tensor_name, meta=tensor.to('meta'), is_inplace=False)
                    unique_live_vars.append(live_tensor)
                    checked_variables.append(live_tensor)
                    all_live_variables.append(live_tensor)

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
            live_var = LiveVariable(name=node.name, meta=meta, is_inplace=is_inplace)
            if not is_inplace:
                unique_live_vars.append(live_var)
            checked_variables.append(live_var)
            all_live_variables.append(live_var)

            # add the model parameters
            if node.op == 'call_module':
                _add_param_or_buf(node, tensor_type='param')
                _add_param_or_buf(node, tensor_type='buffer')

            # add this output variable to the checked list
            checked_variables.append(live_var)

            # check if any input is not checked yet
            for arg in node.args:
                arg_name = str(arg)
                if not checked_variables.exists(arg_name):
                    meta = getattr(node, '_meta_data', None)
                    live_var_from_arg = LiveVariable(name=arg_name, meta=meta, is_inplace=False)
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
            liveness_dict[idx] = stage
        return liveness_dict

    def get_alias_set(self):
        pass
