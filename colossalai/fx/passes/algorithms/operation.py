import math


def _discretize(mem_unit, values):
    return [math.ceil(value / mem_unit) for value in values]


class Chain:

    def __init__(self, fw, bw, cw, cbw, ftmp, btmp, check=True):
        self.fweight = fw
        self.bweight = bw
        self.cweight = cw
        self.cbweight = cbw
        self.fwd_mem_tmp = ftmp
        self.bwd_mem_tmp = btmp
        self.length = len(fw)
        if check and not self.check_lengths():
            raise AttributeError("In Chain, input lists do not have consistent lengths")

    def check_lengths(self):
        return ((len(self.fweight) == self.length) and (len(self.bweight) == self.length + 1)
                and (len(self.cweight) == self.length + 1) and (len(self.fwd_mem_tmp) == self.length)
                and (len(self.bwd_mem_tmp) == self.length + 1) and (len(self.cbweight) == self.length + 1))

    def __repr__(self):
        chain_list = []
        for i in range(self.length):
            chain_list.append((self.fweight[i], self.bweight[i], self.cweight[i], self.cbweight[i], self.fwd_mem_tmp[i],
                               self.bwd_mem_tmp[i]))
        i = self.length
        chain_list.append((None, self.bweight[i], self.cweight[i], self.cbweight[i], None, self.bwd_mem_tmp[i]))
        return chain_list.__repr__()

    def _discretize(self, mem_unit):
        self.cweight = _discretize(mem_unit, self.cweight)
        self.cbweight = _discretize(mem_unit, self.cbweight)
        self.fwd_mem_tmp = _discretize(mem_unit, self.fwd_mem_tmp)
        self.bwd_mem_tmp = _discretize(mem_unit, self.bwd_mem_tmp)


class Operation:

    def shift(self, value):
        if type(self.index) is tuple:
            self.index = tuple(x + value for x in self.index)
        else:
            self.index += value


class Offload(Operation):

    def __init__(self, index, has_bar=False) -> None:
        super().__init__()
        self.index = index
        self.name = "Off"
        self.has_bar = has_bar
        if self.has_bar:
            self.name += "wBar"

    def __repr__(self):
        return f"{self.name}_{self.index}"


class Prefetch(Operation):

    def __init__(self, index, has_bar=False) -> None:
        super().__init__()
        self.index = index
        self.name = "Pre"
        self.has_bar = has_bar
        if self.has_bar:
            self.name += "wBar"

    def __repr__(self):
        return f"{self.name}_{self.index}"


class Forward(Operation):

    def __init__(self, index):
        self.index = index
        self.name = "F"

    def __repr__(self):
        return "{n}_{i}".format(n=self.name, i=self.index)

    def cost(self, chain: Chain):
        if chain is not None:
            return chain.fweight[self.index]
        else:
            return 1


class ForwardEnable(Forward):

    def __init__(self, index):
        super().__init__(index)
        self.name = "Fe"


class ForwardNograd(Forward):

    def __init__(self, index):
        super().__init__(index)
        self.name = "Fn"


class ForwardCheck(Forward):

    def __init__(self, index):
        super().__init__(index)
        self.name = "CF"


class Forwards(Operation):

    def __init__(self, start, end):
        self.index = (start, end)

    def __repr__(self):
        return "F_{i}->{j}".format(i=self.index[0], j=self.index[1])

    def cost(self, chain: Chain):
        if chain is not None:
            return sum(chain.fweight[self.index[0]:self.index[1] + 1])
        else:
            return (self.index[1] - self.index[0] + 1)


def isForward(op):
    return type(op) is Forward or type(op) is Forwards


class Backward(Operation):

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return "B_{i}".format(i=self.index)

    def cost(self, chain: Chain):
        if chain is not None:
            return chain.bweight[self.index]
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

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return "{n}_{i}".format(n=self.name, i=self.index)

    def cost(self, chain: Chain):
        return 0


class WriteMemory(MemoryAccess):

    def __init__(self, index):
        super().__init__(index)
        self.name = "WM"


class ReadMemory(MemoryAccess):

    def __init__(self, index):
        super().__init__(index)
        self.name = "RM"


class DiscardMemory(MemoryAccess):

    def __init__(self, index):
        super().__init__(index)
        self.name = "DM"


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
