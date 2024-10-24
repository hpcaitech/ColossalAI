import queue

from colossalai.pipeline.stage_manager import PipelineStageManager


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
        # print(f"chunk id {chunk} queue size {cls.weight_grad_queue[chunk].qsize()}")
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

    @classmethod
    def clear(cls, stage_manager: PipelineStageManager, chunk=0):
        pass
        # print(f"stage {stage_manager.stage} len_chunk_0 {cls.weight_grad_queue[0].qsize()} len_chunk_1 {cls.weight_grad_queue[1].qsize()}")
        # while cls.weight_grad_queue[chunk].qsize() > 0:
        #     stored_grads = cls.weight_grad_queue[chunk].get()
        #     for total_input, grad_output, weight, func in stored_grads:
        #         if weight.grad is not None:
        #             func(total_input, grad_output, weight.grad)
        #         # for first bwd; weight.grad is None, assign grad_weight to weight.grad
        #         else:
        #             grad_weight = func(total_input, grad_output)
        #             weight.grad = grad_weight

        # weight_grad_tasks = []
        # while cls.weight_grad_queue[chunk].qsize() > 0:
        #     stored_grads = cls.weight_grad_queue[chunk].get()
        #     if len(weight_grad_tasks) == 0:
        #         for _ in stored_grads:
        #             weight_grad_tasks.append([])
        #     else:
        #         assert len(weight_grad_tasks) == len(stored_grads)
        #     for i, task in enumerate(stored_grads):
        #         weight_grad_tasks[i].append(task)

        # if stage_manager.is_last_stage(ignore_chunk=True) and chunk == 1:
        #     assert len(weight_grad_tasks) > 0
        #     output_layer_grads = weight_grad_tasks[0]
        #     for j in range(len(output_layer_grads)):
        #         total_input, grad_output, weight, func = output_layer_grads[j]
        #         if output_layer_weight is None:
        #             output_layer_weight = weight
        #         assert output_layer_weight is weight
        #         func(total_input, grad_output, weight.grad)
        #         output_layer_grads[j] = None  # release memory
        #     weight_grad_tasks = weight_grad_tasks[1:]

        # for i in range(len(weight_grad_tasks)):
        #     tasks = weight_grad_tasks[i]
        #     param = None
        #     for j in range(len(tasks)):
        #         total_input, grad_output, weight, func = tasks[j]
        #         if param is None:
        #             param = weight
        #         assert param is weight
        #         func(total_input, grad_output, weight.grad)
        #         tasks[j] = None  # release memory
        #     weight_grad_tasks[i] = None  # release memory
