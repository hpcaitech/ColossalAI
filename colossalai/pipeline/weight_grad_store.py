import queue


class WeightGradStore:

    cache = []
    weight_grad_queue = [queue.Queue(), queue.Queue()]

    @classmethod
    def put(cls, total_input, grad_output, weight, func):
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
                if isinstance(weight, tuple):
                    # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias.
                    # View will lead to weight ptr change
                    # weight_cal & weight_origin in tuple, weight_cal use to cal dw, weight_origin use to update
                    _, weight_origin = weight
                    if weight_origin.grad is not None:
                        func(total_input, grad_output, weight_origin.grad)
                    # for first bwd; weight.grad is None, assign grad_weight to weight.grad
                    else:
                        grad_weight = func(total_input, grad_output)
                        weight_origin.grad = grad_weight
                else:
                    if weight.grad is not None:
                        func(total_input, grad_output, weight.grad)
                    # for first bwd; weight.grad is None, assign grad_weight to weight.grad
                    else:
                        grad_weight = func(total_input, grad_output)
                        weight.grad = grad_weight
        else:
            raise Exception("Pop empty queue.")
