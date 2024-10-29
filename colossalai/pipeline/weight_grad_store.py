import queue


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
