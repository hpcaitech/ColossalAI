import torch

from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.gemini.ophooks import BaseOpHook


class MemTracerOpHook(BaseOpHook):
    """
    TODO() what if parameters are sharded by multiple submodules.
    register buff on its father node
    """

    def __init__(self):
        super().__init__()
        self.mem_monitor = SyncCudaMemoryMonitor()
        self._cur_non_model_data_vol = 0
        self._non_model_data_list = []
        self._cur_model_data_vol = 0

    def _move_module_to_dev(self, module, dev: str) -> int:
        """
        move module to target dev
        Args:
            module (torch.nn.Module): a PyTorch module
            dev (torch.device): the target device
        Returns:
            int: the data volume of this module on the cuda
        """
        assert isinstance(dev, str), f"device should be a str not torch.device"
        comm_volume = 0
        for p in module.parameters():
            if p.data.device.type != dev:
                p.data = p.data.to(dev)
                comm_volume += p.data.numel() * p.data.element_size()
            if p.grad is not None:
                if p.grad.device.type != dev:
                    p.grad = p.grad.to(dev)
                    comm_volume += p.grad.numel() * p.grad.element_size()

        for buf in module.buffers():
            if buf.device.type != dev:
                buf.data = buf.data.to(dev)
                comm_volume += buf.data.numel() * buf.data.element_size()

        if dev == 'cuda':
            self._cur_model_data_vol = comm_volume

        return comm_volume

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        if module.training:
            cuda_volume = self.mem_monitor.finish()
            comm_volume = self._move_module_to_dev(module, 'cuda')
            self.mem_monitor.start()
            # print(f'FWD PRE {module.__class__.__name__} cuda used {(cuda_volume) / 1e6} MB')

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        if module.training:
            cuda_volume = self.mem_monitor.finish()
            comm_volume = self._move_module_to_dev(module, 'cpu')
            self._non_model_data_list.append(cuda_volume - comm_volume)
            # print(f'FWD POST {module.__class__.__name__} cuda used {(cuda_volume) / 1e6} MB, non-model data used {(cuda_volume - comm_volume) / 1e6} MB')

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        assert isinstance(module, torch.nn.Module)
        if module.training:
            cuda_volume = self.mem_monitor.finish()
            self._move_module_to_dev(module, 'cuda')
            self.mem_monitor.start()
            # print(f'BWD PRE {module.__class__.__name__}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        # bwd Op will generate grad. comm_volume is grad + data volume on cuda.
        assert isinstance(module, torch.nn.Module)
        if module.training:
            cuda_volume = self.mem_monitor.finish()
            comm_volume = self._move_module_to_dev(module, 'cpu')
            self._non_model_data_list.append(cuda_volume - comm_volume)
            # print(f'BWD POST {module.__class__.__name__} {cuda_volume / 1e6} MB, non-model data used {(cuda_volume - comm_volume) / 1e6} MB')

    def pre_iter(self):
        pass

    def post_iter(self):
        self.mem_monitor.finish()
        # print(f'post_iter')

    def print_non_model_data(self):
        print(self._non_model_data_list)

    def save_results(self, filename):
        self.mem_monitor.save(filename)

    def show_mem_stats(self):
        start_timestamp = min(self.mem_monitor.time_stamps)
        self.mem_monitor.time_stamps = [elem - start_timestamp for elem in self.mem_monitor.time_stamps]
        min_mem_used = min(self.mem_monitor.mem_stats)
        self.mem_monitor.mem_stats = [elem - min_mem_used for elem in self.mem_monitor.mem_stats]
        print(self.mem_monitor.time_stamps)
        print(self.mem_monitor.mem_stats)
