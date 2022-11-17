from colossalai.gemini.ophooks import register_ophooks_recursively
from colossalai.gemini.ophooks.mem_trace_hook import MemTracerOpHook

__all__ = ['MemtracerWrapper']


class _Wrapper():

    def __init__(self, model, ophook_list):
        self._ophook_list = ophook_list
        self._model = model

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

    def backward(self, loss):
        loss.backward()
        for ophook in self._ophook_list:
            ophook.post_iter()

    def save_results(self, filename):
        for ophook in self._ophook_list:
            ophook.save_results(filename)

    def show_mem_stats(self):
        self._ophook_list[0].show_mem_stats()


def MemtracerWrapper(model):
    ophook_list = [MemTracerOpHook()]
    register_ophooks_recursively(model, ophook_list)
    engine = _Wrapper(model, ophook_list)
    return engine
