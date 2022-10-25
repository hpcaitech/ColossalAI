import math
from abc import ABC
from typing import List

from torch.utils._pytree import tree_map


class Chain:

    def __init__(self,
                 ftime: List[float],
                 btime: List[float],
                 x: List[int],
                 xbar: List[int],
                 ftmp: List[int],
                 btmp: List[int],
                 check_consistency: bool = True):
        """The chain is a basic linearized structure for solving the dynamic programming problem for activation checkpoint.
        See paper https://hal.inria.fr/hal-02352969 for details.

        Args:
            ftime (List[float]): The forward time of each node.
            btime (List[float]): The backward time of each node.
            x (List[int]): The forward memory of each node (if save_output). Same as `a` in the paper.
            xbar (List[int]): The forward memory of each node (if save_all). Same as `a_bar` in the paper.
            ftmp (List[int]): The temporary forward memory of each node.
            btmp (List[int]): The temporary backward memory of each node, can be used to control memory budget.
            check_consistency (bool, optional): Check the lengths consistency for the `Chain`. Defaults to True.
        """
        self.ftime = ftime
        self.btime = btime
        self.x = x
        self.xbar = xbar
        self.ftmp = ftmp
        self.btmp = btmp
        self.length = len(ftime)
        if check_consistency and not self.check_lengths():
            raise AttributeError("In Chain, input lists do not have consistent lengths")

    def check_lengths(self):
        return ((len(self.ftime) == self.length) and (len(self.btime) == self.length + 1)
                and (len(self.x) == self.length + 1) and (len(self.ftmp) == self.length)
                and (len(self.btmp) == self.length + 1) and (len(self.xbar) == self.length + 1))

    def __repr__(self):
        chain_list = []
        for i in range(self.length):
            chain_list.append((self.ftime[i], self.btime[i], self.x[i], self.xbar[i], self.ftmp[i], self.btmp[i]))
        i = self.length
        chain_list.append((None, self.btime[i], self.x[i], self.xbar[i], None, self.btmp[i]))
        return chain_list.__repr__()

    def discretize_all(self, unit: int):
        """Discretize the chain into a list of chains according to unit size."""
        discretizer = lambda val: math.ceil(val / unit)
        self.x = tree_map(discretizer, self.x)
        self.xbar = tree_map(discretizer, self.xbar)
        self.ftmp = tree_map(discretizer, self.ftmp)
        self.btmp = tree_map(discretizer, self.btmp)


class Operation(ABC):
    name = "Op"

    def __repr__(self) -> str:
        return f"{self.name}_{self.index}"

    def shift(self, value):
        if type(self.index) is tuple:
            self.index = tuple(x + value for x in self.index)
        else:
            self.index += value


class Forward(Operation):
    name = "F"

    def __init__(self, index):
        self.index = index

    def cost(self, chain: Chain):
        if chain is not None:
            return chain.ftime[self.index]
        else:
            return 1


class ForwardEnable(Forward):
    name = "Fe"


class ForwardNograd(Forward):
    name = "Fn"


class ForwardCheck(Forward):
    name = "CF"


class Forwards(Operation):

    def __init__(self, start, end):
        self.index = (start, end)

    def __repr__(self):
        return "F_{i}->{j}".format(i=self.index[0], j=self.index[1])

    def cost(self, chain: Chain):
        if chain is not None:
            return sum(chain.ftime[self.index[0]:self.index[1] + 1])
        else:
            return (self.index[1] - self.index[0] + 1)


def isForward(op):
    return type(op) is Forward or type(op) is Forwards


class Backward(Operation):
    name = "B"

    def __init__(self, index):
        self.index = index

    def cost(self, chain: Chain):
        if chain is not None:
            return chain.btime[self.index]
        else:
            return 1


class Loss(Operation):

    def __init__(self):
        pass

    def __repr__(self):
        return "L"

    def cost(self, chain):
        return 0


class MemoryAccess(Operation):
    name = "MA"

    def __init__(self, index):
        self.index = index

    def cost(self, chain: Chain):
        return 0


class WriteMemory(MemoryAccess):
    name = "WM"


class ReadMemory(MemoryAccess):
    name = "RM"


class DiscardMemory(MemoryAccess):
    name = "DM"


class Function:

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.str_args = ','.join(str(v) for v in self.args)

    def __repr__(self):
        return "{n}({args})".format(n=self.name, args=self.str_args)


class Sequence:

    def __init__(self, function):
        self.sequence = []    #List of Operation and Sequence
        self.function = function    #Description the function (name and parameters)

    def __repr__(self):
        return repr(self.list_operations())

    def list_operations(self):
        op_list = []
        for x in self.sequence:
            if isinstance(x, Operation):
                op_list.append(x)
            else:
                assert isinstance(x, Sequence)
                op_list += x.list_operations()
        return op_list

    def insert(self, operation):
        self.sequence.append(operation)

    def remove(self, operation_index):
        del self.sequence[operation_index]

    def insert_sequence(self, sequence):
        self.sequence.append(sequence)

    def shift(self, value):
        for x in self.sequence:
            x.shift(value)
        return self

    def remove_useless_write(self):
        if self.sequence:
            if isinstance(self.sequence[0], WriteMemory):
                self.remove(0)
        return self

    def get_makespan(self, chain):
        return sum(op.cost(chain) for op in self.list_operations())

    def without_suffix(self):
        ops = self.list_operations()
        end_of_first_phase = [i for i in range(len(ops)) if type(ops[i]) is Loss][0]
        try:
            last_idx = max(i for i in range(end_of_first_phase) if not type(ops[i]) is ForwardEnable)
        except ValueError:
            last_idx = -1
        if last_idx == end_of_first_phase - 1:
            return (self, None)
        chain_length = ops[end_of_first_phase -
                           1].index    ## Some assumption here about the sequence (finishes with Forward_L
        start_of_fwd_enable_chain = ops[last_idx + 1].index    ## And starts with B_L), but should be fine in practice
        result = Sequence(Function("Strip", self.function.name, *self.function.args, start_of_fwd_enable_chain))
        for i in range(last_idx + 1):
            result.insert(ops[i])
        result.insert(Loss())
        for i in range(chain_length, start_of_fwd_enable_chain - 1, -1):
            position = end_of_first_phase + 1 + (chain_length - i)
            assert type(ops[position]) is Backward
            assert ops[position].index == i
        for i in range(end_of_first_phase + 1 + 1 + chain_length - start_of_fwd_enable_chain, len(ops)):
            result.insert(ops[i])
        return (result, start_of_fwd_enable_chain)
