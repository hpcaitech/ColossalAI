import torch

from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.gemini.ophooks import BaseOpHook


class MemTracerOpHook(BaseOpHook):

    def __init__(self):
        super().__init__()
        self.mem_monitor = SyncCudaMemoryMonitor()
        self._cur_non_model_data_vol = 0
        self._non_model_data_list = []

    def _move_module_to_dev(self, module, dev) -> int:
        """_move_module_to_dev
        move module to cuda
        Args:
            module (_type_): _description_
            dev (_type_): _description_
        Returns:
            int: the communication volume
        """
        comm_volume = 0
        for p in module.parameters():
            if p.data.device != dev:
                p.data = p.data.to(dev)
                comm_volume += p.data.numel() * p.data.element_size()
            if p.grad is not None:
                if p.grad.device != dev:
                    p.grad = p.grad.to(dev)
                    comm_volume += p.grad.numel() * p.grad.element_size()
        return comm_volume

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        if module.training:
            cuda_volume = self.mem_monitor.finish()
            comm_volume = self._move_module_to_dev(module, 'cuda')
            self.mem_monitor.start()
            # print(f'FWD PRE {module.__class__.__name__}')

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        if module.training:
            comm_volume = self._move_module_to_dev(module, 'cpu')
            cuda_volume = self.mem_monitor.finish()
            print(f'FWD POST {module.__class__.__name__} {(cuda_volume - comm_volume) /1e6} MB')

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            cuda_volume = self.mem_monitor.finish()
            comm_volume = self._move_module_to_dev(module, 'cuda')
            self.mem_monitor.start()
            # print(f'BWD PRE {module.__class__.__name__}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            cuda_volume = self.mem_monitor.finish()
            comm_volume = self._move_module_to_dev(module, 'cpu')
            print(f'BWD POST {module.__class__.__name__} {(cuda_volume - comm_volume) /1e6} MB')

    def pre_iter(self):
        pass

    def post_iter(self):
        self.mem_monitor.finish()
        # print(f'post_iter')

    def save_results(self, filename):
        self.mem_monitor.save(filename)

    def show_mem_stats(self):
        start_timestamp = min(self.mem_monitor.time_stamps)
        self.mem_monitor.time_stamps = [elem - start_timestamp for elem in self.mem_monitor.time_stamps]
        min_mem_used = min(self.mem_monitor.mem_stats)
        self.mem_monitor.mem_stats = [elem - min_mem_used for elem in self.mem_monitor.mem_stats]
        print(self.mem_monitor.time_stamps)
        print(self.mem_monitor.mem_stats)
