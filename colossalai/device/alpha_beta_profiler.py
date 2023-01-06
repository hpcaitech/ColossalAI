import math
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from colossalai.logging import get_dist_logger

GB = int((1 << 30))
BYTE = 4
FRAMEWORK_LATENCY = 0


class AlphaBetaProfiler:
    '''
    Profile alpha and beta value for a given device list.

    Usage:
        # Note: the environment of execution is supposed to be
        # multi-process with multi-gpu in mpi style.
        >>> physical_devices = [0, 1, 4, 5]
        >>> ab_profiler = AlphaBetaProfiler(physical_devices)
        >>> ab_dict = profiler.profile_ab()
        >>> print(ab_dict)
        {(0, 1): (1.9641406834125518e-05, 4.74049549614719e-12), (0, 4): (1.9506998360157013e-05, 6.97421973297474e-11), (0, 5): (2.293858677148819e-05, 7.129930361393644e-11),
         (1, 4): (1.9010603427886962e-05, 7.077968863788975e-11), (1, 5): (1.9807778298854827e-05, 6.928845708992215e-11), (4, 5): (1.8681809306144713e-05, 4.7522367291330524e-12),
         (1, 0): (1.9641406834125518e-05, 4.74049549614719e-12), (4, 0): (1.9506998360157013e-05, 6.97421973297474e-11), (5, 0): (2.293858677148819e-05, 7.129930361393644e-11),
         (4, 1): (1.9010603427886962e-05, 7.077968863788975e-11), (5, 1): (1.9807778298854827e-05, 6.928845708992215e-11), (5, 4): (1.8681809306144713e-05, 4.7522367291330524e-12)}
    '''

    def __init__(self,
                 physical_devices: List[int],
                 ctype: str = 'a',
                 warmup: int = 5,
                 repeat: int = 25,
                 latency_iters: int = 5):
        '''
        Args:
            physical_devices: A list of device id, each element inside it is the global rank of that device.
            ctype: 'a' for all-reduce, 'b' for broadcast.
            warmup: Number of warmup iterations.
            repeat: Number of iterations to measure.
            latency_iters: Number of iterations to measure latency.
        '''
        self.physical_devices = physical_devices
        self.ctype = ctype
        self.world_size = len(physical_devices)
        self.warmup = warmup
        self.repeat = repeat
        self.latency_iters = latency_iters
        self.process_group_dict = None
        self._init_profiling()

    def _init_profiling(self):
        # Create process group list based on its global rank
        process_group_list = []
        for f_index in range(self.world_size - 1):
            for b_index in range(f_index + 1, self.world_size):
                process_group_list.append((self.physical_devices[f_index], self.physical_devices[b_index]))

        # Create process group dict which maps process group to its handler
        process_group_dict = {}
        for process_group in process_group_list:
            pg_handler = dist.new_group(process_group)
            process_group_dict[process_group] = pg_handler

        self.process_group_dict = process_group_dict

    def _profile(self, process_group, pg_handler, nbytes):
        logger = get_dist_logger()
        rank = dist.get_rank()
        src_device_num = process_group[0]
        world_size = len(process_group)

        device = torch.cuda.current_device()
        buf = torch.randn(nbytes // 4).to(device)

        torch.cuda.synchronize()
        # warmup
        for _ in range(self.warmup):
            if self.ctype == "a":
                dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=pg_handler)
            elif self.ctype == "b":
                dist.broadcast(buf, src=src_device_num, group=pg_handler)
        torch.cuda.synchronize()

        dist.barrier(group=pg_handler)
        begin = time.perf_counter()
        for _ in range(self.repeat):
            if self.ctype == "a":
                dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=pg_handler)
            elif self.ctype == "b":
                dist.broadcast(buf, src=src_device_num, group=pg_handler)
        torch.cuda.synchronize()
        end = time.perf_counter()
        dist.barrier(group=pg_handler)

        if rank == src_device_num:
            avg_time_s = (end - begin) / self.repeat - FRAMEWORK_LATENCY
            alg_band = nbytes / avg_time_s
            if self.ctype == "a":
                # convert the bandwidth of all-reduce algorithm to the bandwidth of the hardware.
                bus_band = 2 * (world_size - 1) / world_size * alg_band
                bus_band = alg_band
            elif self.ctype == "b":
                bus_band = alg_band

            logger.info(
                f"GPU:{rank}, Bytes: {nbytes} B,Time: {round(avg_time_s * 1e6,2)} us, Bus bandwidth: {round(bus_band / GB,2)} GB/s"
            )
            return (avg_time_s, alg_band)
        else:
            # Just a placeholder
            return (None, None)

    def profile_latency(self, process_group, pg_handler):
        '''
        This function is used to profile the latency of the given process group with a series of bytes.

        Args:
            process_group: A tuple of global rank of the process group.
            pg_handler: The handler of the process group.

        Returns:
            latency: None if the latency is not measured, otherwise the median of the latency_list.
        '''
        latency_list = []
        for i in range(self.latency_iters):
            nbytes = int(BYTE << i)
            (t, _) = self._profile(process_group, pg_handler, nbytes)
            latency_list.append(t)

        if latency_list[0] is None:
            latency = None
        else:
            median_index = math.floor(self.latency_iters / 2)
            latency = latency_list[median_index]

        return latency

    def profile_bandwidth(self, process_group, pg_handler, maxbytes):
        '''
        This function is used to profile the bandwidth of the given process group.

        Args:
            process_group: A tuple of global rank of the process group.
            pg_handler: The handler of the process group.
        '''
        (_, bandwidth) = self._profile(process_group, pg_handler, maxbytes)
        return bandwidth

    def profile_ab(self):
        '''
        This method is used to profiling the alpha and beta value for a given device list.

        Returns:
            alpha_beta_dict: A dict which maps process group to its alpha and beta value.
        '''
        alpha_beta_dict: Dict[Tuple[int], Tuple[float]] = {}
        rank = dist.get_rank()

        def get_max_nbytes(process_group: Tuple[int], pg_handler: dist.ProcessGroup):
            assert rank in process_group
            device = torch.cuda.current_device()
            rank_max_nbytes = torch.cuda.mem_get_info(device)[0]
            rank_max_nbytes = torch.tensor(rank_max_nbytes, device=device)
            dist.all_reduce(rank_max_nbytes, op=dist.ReduceOp.MIN, group=pg_handler)
            max_nbytes = min(int(1 * GB), int(GB << int(math.log2(rank_max_nbytes.item() / GB))))
            return max_nbytes

        for process_group, pg_handler in self.process_group_dict.items():
            if rank not in process_group:
                max_nbytes = None
                alpha = None
                bandwidth = None
            else:
                max_nbytes = get_max_nbytes(process_group, pg_handler)
                alpha = self.profile_latency(process_group, pg_handler)
                bandwidth = self.profile_bandwidth(process_group, pg_handler, maxbytes=max_nbytes)

            if bandwidth is None:
                beta = None
            else:
                beta = 1 / bandwidth

            broadcast_list = [alpha, beta]
            dist.broadcast_object_list(broadcast_list, src=process_group[0])
            alpha_beta_dict[process_group] = tuple(broadcast_list)

        # add symmetry pair to the apha_beta_dict
        symmetry_ab_dict = {}
        for process_group, alpha_beta_pair in alpha_beta_dict.items():
            symmetry_process_group = (process_group[1], process_group[0])
            symmetry_ab_dict[symmetry_process_group] = alpha_beta_pair

        alpha_beta_dict.update(symmetry_ab_dict)

        return alpha_beta_dict
