import math
from abc import ABC
from typing import Any, Iterable, List

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
        if check_consistency and not self.check_lengths():
            raise AttributeError("In Chain, input lists do not have consistent lengths")

    def check_lengths(self):
        return ((len(self.ftime) == len(self)) and (len(self.btime) == len(self) + 1) and (len(self.x) == len(self) + 1)
                and (len(self.ftmp) == len(self)) and (len(self.btmp) == len(self) + 1)
                and (len(self.xbar) == len(self) + 1))

    def __repr__(self):
        chain_list = []
        for i in range(len(self)):
            chain_list.append((self.ftime[i], self.btime[i], self.x[i], self.xbar[i], self.ftmp[i], self.btmp[i]))
        i = len(self)
        chain_list.append((None, self.btime[i], self.x[i], self.xbar[i], None, self.btmp[i]))
        return chain_list.__repr__()

    def __len__(self):
        return len(self.ftime)

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


class Sequence(list):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return repr(self.list_operations())

    def list_operations(self):
        op_list = []
        for x in self:
            if isinstance(x, Operation):
                op_list.append(x)
            else:
                assert isinstance(x, Sequence)
                op_list += x.list_operations()
        return op_list
