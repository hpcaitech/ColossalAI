import queue

# from megatron import get_args
# from megatron.core import parallel_state
# from megatron.core.distributed.finalize_model_grads import _allreduce_embedding_grads
# from megatron.core.utils import get_model_config, get_attr_wrapped_model


class WeightGradStore:

    cache = []
    weight_grad_queue = [queue.Queue(), queue.Queue()]

    @classmethod
    def put(cls, total_input, grad_output, weight, func):
        # func(total_input, grad_output, weight.main_grad)
        cls.cache.append((total_input, grad_output, weight, func))

    @classmethod
    def flush(cls, chunk=0):
        cls.weight_grad_queue[chunk].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0):
        if cls.weight_grad_queue[chunk].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk].get()
            for total_input, grad_output, weight, func in stored_grads:
                if weight.grad is not None:
                    func(total_input, grad_output, weight.grad)
                # for first bwd; weight.grad is None, assign grad_weight to weight.grad
                else:
                    grad_weight = func(total_input, grad_output)
                    weight.grad = grad_weight
        else:
            raise Exception("Pop empty queue.")

    # @classmethod
    # def clear(cls, model, chunk=0):
    #     weight_grad_tasks = []
    #     while cls.weight_grad_queue[chunk].qsize() > 0:
    #         stored_grads = cls.weight_grad_queue[chunk].get()
    #         if len(weight_grad_tasks) == 0:
    #             for _ in stored_grads:
    #                 weight_grad_tasks.append([])
    #         else:
    #             assert len(weight_grad_tasks) == len(stored_grads)
    #         for i, task in enumerate(stored_grads):
    #             weight_grad_tasks[i].append(task)
    #     weight_params = []
    #     handles = []
    #     if get_args().overlap_grad_reduce:
    #         handles += model.async_reduce_grad()

    #     output_layer_weight = None
    #     if parallel_state.is_pipeline_last_stage():
    #         assert len(weight_grad_tasks) > 0
    #         output_layer_grads = weight_grad_tasks[0]
    #         for j in range(len(output_layer_grads)):
    #             total_input, grad_output, weight, func = output_layer_grads[j]
    #             if output_layer_weight is None:
    #                 output_layer_weight = weight
    #             assert output_layer_weight is weight
    #             func(total_input, grad_output, weight.main_grad)
    #             output_layer_grads[j] = None  # release memory
    #         weight_grad_tasks = weight_grad_tasks[1:]
    #         if get_args().overlap_grad_reduce:
    #             handles += model.async_reduce_grad(output_layer_weight)

    #     if parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage():
    #         model_module = get_attr_wrapped_model(model, 'pre_process', return_model_obj=True)
    #         if model_module.share_embeddings_and_output_weights:
    #             # if share_embeddings_and_output_weights, wait all-reduce for embeddings
    #             for handle in handles:
    #                 if handle is not None:
    #                     handle.wait()
    #             handles = []

    #     config = get_model_config(model)
    #     # Do async all-reduce for embedding grads firstly, so that the rank 0 won't
    #     # be blocked
    #     embedding_handles = _allreduce_embedding_grads([model], config, async_op=True)
    #     handles += embedding_handles

    #     for i in range(len(weight_grad_tasks)):
    #         tasks = weight_grad_tasks[i]
    #         param = None
    #         for j in range(len(tasks)):
    #             total_input, grad_output, weight, func = tasks[j]
    #             if param is None:
    #                 param = weight
    #             assert param is weight
    #             assert not (weight is output_layer_weight)
    #             func(total_input, grad_output, weight.main_grad)
    #             tasks[j] = None  # release memory
    #         weight_params.append(param)
    #         if get_args().overlap_grad_reduce:
    #             # All-reduce param grad here
    #             handles += model.async_reduce_grad(param)
    #         weight_grad_tasks[i] = None  # release memory

    #     # timers('wait_all_reduce', log_level=1).start(barrier=False)
    #     for handle in embedding_handles:
    #         if handle is not None:
    #             handle.wait()
    #     # timers('wait_all_reduce').stop()
