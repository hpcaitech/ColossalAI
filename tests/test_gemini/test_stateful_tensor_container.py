import pytest
import torch

from colossalai.gemini.stateful_tensor import TensorState, StatefulTensor
from colossalai.gemini.stateful_tensor_container import QueueSTContainer, HeapSTContainer


@pytest.mark.dist
def test_stateful_tensor_container():
    st1 = StatefulTensor(torch.randn(1, device='cuda'))
    st2 = StatefulTensor(torch.randn(2, device='cuda'))
    st3 = StatefulTensor(torch.randn(3, device='cuda'))
    stateful_tensor_list = [st1, st2, st3]
    step_list = [st1, st2, st3, st3, st2, st1]

    compute_step_dict = dict()
    compute_step_dict[st1] = [0, 5]
    compute_step_dict[st2] = [1, 4]
    compute_step_dict[st3] = [2, 3]

    def run_queue_test():
        # test queue container
        queue_container = QueueSTContainer(compute_step_dict, 6)
        queue_container.create(stateful_tensor_list)

        res_list = []

        for i in range(6):
            stateful_tensor = step_list[i]
            stateful_tensor.trans_state(TensorState.COMPUTE)
            st_out = queue_container.pop()
            st_out.move_to(torch.device('cpu'))

            res_list.append(st_out.payload.size(0))

            stateful_tensor.move_to(torch.device('cuda'))
            queue_container.push(stateful_tensor, i)
            stateful_tensor.trans_state(TensorState.HOLD)

        assert res_list == [2, 3, 1, 2, 3, 2]

    run_queue_test()

    def run_heap_test():
        # test heap container
        st1.move_to(torch.device('cuda'))
        st2.move_to(torch.device('cuda'))
        st3.move_to(torch.device('cuda'))

        heap_container = HeapSTContainer(compute_step_dict, 6)
        heap_container.create(stateful_tensor_list)

        res_list = []

        for i in range(6):
            stateful_tensor = step_list[i]
            stateful_tensor.trans_state(TensorState.COMPUTE)
            st_out = heap_container.pop()

            if st_out is not None:
                res_list.append(st_out.payload.size(0))
                st_out.move_to(torch.device('cpu'))

            stateful_tensor.move_to(torch.device('cuda'))
            heap_container.push(stateful_tensor, i)
            stateful_tensor.trans_state(TensorState.HOLD)

        assert res_list == [3, 1, 2, 3, 2]

    run_heap_test()


if __name__ == '__main__':
    test_stateful_tensor_container()
