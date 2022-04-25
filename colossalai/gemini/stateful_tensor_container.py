import queue
import heapq
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from colossalai.gemini.stateful_tensor import StatefulTensor, TensorState


def evict_check(st: StatefulTensor) -> bool:
    if st.state is not TensorState.COMPUTE and st.device.type == 'cuda':
        return True
    return False


# Here ST means Stateful Tensor
class BaseSTContainer(ABC):
    """A type of container that store all potential stateful tensors which can be evicted from
    CUDA. This kind of stateful tensor should satisfy two conditions. One is that it hasn't been
    evicted, meaning the type of its device is CUDA, the other is that it isn't pinned in CUDA
    memory, meaning its state isn't COMPUTE.

    This container should get a stateful tensor when it become HOLD_LIKE from COMPUTE.
    And it pops stateful tensors in function, `evict_tensors`.

    In order to acquire an optimal eviction policy, users may need to offer computation step
    index of each stateful tensor. So we can use a heap to maintain all potential evictable
    statefule tensors. When poping, we can get the stateful tensor that used furthest in
    current computation step.
    """

    def __init__(self, compute_step_dict: Dict[StatefulTensor, List[int]], total_step: int):
        self.compute_step_dict = compute_step_dict
        self.total_step = total_step

    @abstractmethod
    def empty(self) -> bool:
        pass

    @abstractmethod
    def create(self, stateful_tensor_list: List[StatefulTensor]) -> None:
        pass

    @abstractmethod
    def push(self, stateful_tensor: StatefulTensor, cur_step: int) -> None:
        pass

    @abstractmethod
    def pop(self) -> Optional[StatefulTensor]:
        pass


class QueueSTContainer(BaseSTContainer):
    """Queue type stateful tensor container. This is used in 'cpu' tensor placement policy.
    It pops potential evictable stateful tensors in FIFO.
    """

    def __init__(self, compute_step_dict: Dict[StatefulTensor, List[int]], total_step: int):
        super().__init__(compute_step_dict, total_step)
        self.container = None

    def empty(self) -> bool:
        assert self.container is not None
        return self.container.empty()

    def create(self, stateful_tensor_list: List[StatefulTensor]) -> None:
        self.container = queue.SimpleQueue()
        for stateful_tensor in stateful_tensor_list:
            self.container.put(stateful_tensor)

    def push(self, stateful_tensor: StatefulTensor, cur_step: int) -> None:
        self.container.put(stateful_tensor)

    def pop(self) -> Optional[StatefulTensor]:
        ret = None
        while not self.empty():
            out_tensor = self.container.get()
            if evict_check(out_tensor):
                ret = out_tensor
                break

        return ret


class HeapSTContainer(BaseSTContainer):
    """Heap type stateful tensor container. This is used in 'auto' tensor placement policy.
    It pops potential evictable stateful tensors in the order of the distance between current
    step and next used step.
    """

    def __init__(self, compute_step_dict: Dict[StatefulTensor, List[int]], total_step: int):
        super().__init__(compute_step_dict, total_step)
        self.container = None

    def empty(self) -> bool:
        assert self.container is not None
        return self.container == []

    def create(self, stateful_tensor_list: List[StatefulTensor]) -> None:
        self.container = []
        for stateful_tensor in stateful_tensor_list:
            # we want to pop the tensor which has the greatest next_step
            # so the weight is next_step multiplied by -1
            weight = -self.__get_next_compute_step(stateful_tensor, -1)
            self.container.append((weight, stateful_tensor))
        heapq.heapify(self.container)

    def push(self, stateful_tensor: StatefulTensor, cur_step: int) -> None:
        # we want to pop the tensor which has the greatest next_step
        # so the weight is next_step multiplied by -1
        weight = -self.__get_next_compute_step(stateful_tensor, cur_step)
        heapq.heappush(self.container, (weight, stateful_tensor))

    def pop(self) -> Optional[StatefulTensor]:
        ret = None
        while not self.empty():
            _, out_tensor = heapq.heappop(self.container)
            if evict_check(out_tensor):
                ret = out_tensor
                break
        return ret

    def __get_next_compute_step(self, stateful_tensor: StatefulTensor, cur_step: int):
        # compute the id of next step
        # if the tensor is not used in the furture
        # next_step is set to the maximum
        next_step = self.total_step
        step_list = self.compute_step_dict[stateful_tensor]
        for step in step_list:
            if step > cur_step:
                next_step = step
                break
        return next_step
