from async_memtracer import AsyncMemoryMonitor
import torch

if __name__ == '__main__':
    async_mem_monitor = AsyncMemoryMonitor()
    input = torch.randn(2, 20).cuda()
    OP1 = torch.nn.Linear(20, 30).cuda()
    OP2 = torch.nn.Linear(30, 40).cuda()

    async_mem_monitor.start()
    output = OP1(input)
    async_mem_monitor.finish()
    async_mem_monitor.start()
    output = OP2(output)
    async_mem_monitor.finish()
    async_mem_monitor.save('log.pkl')
