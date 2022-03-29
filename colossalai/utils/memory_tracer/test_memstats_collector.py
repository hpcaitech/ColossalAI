from colossalai.utils.memory_tracer.memstats_collector import MemStatsCollector
import torch


def test_mem_collector():
    collector = MemStatsCollector()

    collector.start_collection()

    a = torch.randn(10).cuda()

    # sampling at time 0
    collector.sample_memstats()

    m_a = torch.randn(10).cuda()
    b = torch.randn(10).cuda()

    # sampling at time 1
    collector.sample_memstats()

    a = b

    # sampling at time 2
    collector.sample_memstats()

    collector.finish_collection()
    collector.reset_sampling_cnter()

    # do nothing after collection, just advance sampling cnter
    collector.sample_memstats()
    collector.sample_memstats()

    cuda_use, overall_use = collector.fetch_memstats()
    print(cuda_use, overall_use)

    print(collector.overall_cuda)


if __name__ == '__main__':
    test_mem_collector()
