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
    """
    Profile alpha and beta value for a given device list.

    Usage:
        # Note: the environment of execution is supposed to be
        # multi-process with multi-gpu in mpi style.
        >>> physical_devices = [0, 1, 4, 5]
        >>> ab_profiler = AlphaBetaProfiler(physical_devices)
        >>> ab_dict = profiler.alpha_beta_dict
        >>> print(ab_dict)
        {(0, 1): (1.9641406834125518e-05, 4.74049549614719e-12), (0, 4): (1.9506998360157013e-05, 6.97421973297474e-11), (0, 5): (2.293858677148819e-05, 7.129930361393644e-11),
         (1, 4): (1.9010603427886962e-05, 7.077968863788975e-11), (1, 5): (1.9807778298854827e-05, 6.928845708992215e-11), (4, 5): (1.8681809306144713e-05, 4.7522367291330524e-12),
         (1, 0): (1.9641406834125518e-05, 4.74049549614719e-12), (4, 0): (1.9506998360157013e-05, 6.97421973297474e-11), (5, 0): (2.293858677148819e-05, 7.129930361393644e-11),
         (4, 1): (1.9010603427886962e-05, 7.077968863788975e-11), (5, 1): (1.9807778298854827e-05, 6.928845708992215e-11), (5, 4): (1.8681809306144713e-05, 4.7522367291330524e-12)}
    """

    def __init__(
        self,
        physical_devices: List[int],
        alpha_beta_dict: Dict[Tuple[int, int], Tuple[float, float]] = None,
        ctype: str = "a",
        warmup: int = 5,
        repeat: int = 25,
        latency_iters: int = 5,
        homogeneous_tolerance: float = 0.1,
    ):
        """
        Args:
            physical_devices: A list of device id, each element inside it is the global rank of that device.
            alpha_beta_dict: A dict which maps a process group to alpha-beta value pairs.
            ctype: 'a' for all-reduce, 'b' for broadcast.
            warmup: Number of warmup iterations.
            repeat: Number of iterations to measure.
            latency_iters: Number of iterations to measure latency.
        """
        self.physical_devices = physical_devices
        self.ctype = ctype
        self.world_size = len(physical_devices)
        self.warmup = warmup
        self.repeat = repeat
        self.latency_iters = latency_iters
        self.homogeneous_tolerance = homogeneous_tolerance
        self.process_group_dict = None
        self._init_profiling()
        if alpha_beta_dict is None:
            self.alpha_beta_dict = self.profile_ab()
        else:
            self.alpha_beta_dict = alpha_beta_dict

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
        """
        This function is used to profile the latency of the given process group with a series of bytes.

        Args:
            process_group: A tuple of global rank of the process group.
            pg_handler: The handler of the process group.

        Returns:
            latency: None if the latency is not measured, otherwise the median of the latency_list.
        """
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

    def profile_bandwidth(self, process_group, pg_handler, maxbytes=(1 * GB)):
        """
        This function is used to profile the bandwidth of the given process group.

        Args:
            process_group: A tuple of global rank of the process group.
            pg_handler: The handler of the process group.
        """
        (_, bandwidth) = self._profile(process_group, pg_handler, maxbytes)
        return bandwidth

    def profile_ab(self):
        """
        This method is used to profiling the alpha and beta value for a given device list.

        Returns:
            alpha_beta_dict: A dict which maps process group to its alpha and beta value.
        """
        alpha_beta_dict: Dict[Tuple[int], Tuple[float]] = {}
        rank = dist.get_rank()
        dist.new_group(self.physical_devices)

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

        # add symmetry pair to the alpha_beta_dict
        symmetry_ab_dict = {}
        for process_group, alpha_beta_pair in alpha_beta_dict.items():
            symmetry_process_group = (process_group[1], process_group[0])
            symmetry_ab_dict[symmetry_process_group] = alpha_beta_pair

        alpha_beta_dict.update(symmetry_ab_dict)

        return alpha_beta_dict

    def search_best_logical_mesh(self):
        """
        This method is used to search the best logical mesh for the given device list.

        The best logical mesh is searched in following steps:
            1. detect homogeneous device groups, we assume that the devices in the alpha_beta_dict
                are homogeneous if the beta value is close enough.
            2. Find the best homogeneous device group contains all the physical devices. The best homogeneous
                device group means the lowest beta value in the groups which contains all the physical devices.
                And the reason we require the group contains all the physical devices is that the devices not in
                the group will decrease the bandwidth of the group.
            3. If the best homogeneous device group is found, we will construct the largest ring for each device
                based on the best homogeneous device group, and the best logical mesh will be the union of all the
                rings. Otherwise, the best logical mesh will be the balanced logical mesh, such as shape (2, 2) for
                4 devices.

        Returns:
            best_logical_mesh: The best logical mesh for the given device list.

        Usage:
            >>> physical_devices = [0, 1, 2, 3]
            >>> ab_profiler = AlphaBetaProfiler(physical_devices)
            >>> best_logical_mesh = profiler.search_best_logical_mesh()
            >>> print(best_logical_mesh)
            [[0, 1], [2, 3]]
        """

        def _power_of_two(integer):
            return integer & (integer - 1) == 0

        def _detect_homogeneous_device(alpha_beta_dict):
            """
            This function is used to detect whether the devices in the alpha_beta_dict are homogeneous.

            Note: we assume that the devices in the alpha_beta_dict are homogeneous if the beta value
                of the devices are in range of [(1 - self.homogeneous_tolerance), (1 + self.homogeneous_tolerance)]
                * base_beta.
            """
            homogeneous_device_dict: Dict[float, List[Tuple[int]]] = {}
            for process_group, (_, beta) in alpha_beta_dict.items():
                if homogeneous_device_dict is None:
                    homogeneous_device_dict[beta] = []
                    homogeneous_device_dict[beta].append(process_group)

                match_beta = None
                for beta_value in homogeneous_device_dict.keys():
                    if beta <= beta_value * (1 + self.homogeneous_tolerance) and beta >= beta_value * (
                        1 - self.homogeneous_tolerance
                    ):
                        match_beta = beta_value
                        break

                if match_beta is not None:
                    homogeneous_device_dict[match_beta].append(process_group)
                else:
                    homogeneous_device_dict[beta] = []
                    homogeneous_device_dict[beta].append(process_group)

            return homogeneous_device_dict

        def _check_contain_all_devices(homogeneous_group: List[Tuple[int]]):
            """
            This function is used to check whether the homogeneous_group contains all physical devices.
            """
            flatten_mesh = []
            for process_group in homogeneous_group:
                flatten_mesh.extend(process_group)
            non_duplicated_flatten_mesh = set(flatten_mesh)
            return len(non_duplicated_flatten_mesh) == len(self.physical_devices)

        def _construct_largest_ring(homogeneous_group: List[Tuple[int]]):
            """
            This function is used to construct the largest ring in the homogeneous_group for each rank.
            """
            # Construct the ring
            ring = []
            ranks_in_ring = []
            for rank in self.physical_devices:
                if rank in ranks_in_ring:
                    continue
                stable_status = False
                ring_for_rank = []
                ring_for_rank.append(rank)
                check_rank_list = [rank]
                rank_to_check_list = []

                while not stable_status:
                    stable_status = True
                    check_rank_list.extend(rank_to_check_list)
                    rank_to_check_list = []
                    for i in range(len(check_rank_list)):
                        check_rank = check_rank_list.pop()
                        for process_group in homogeneous_group:
                            if check_rank in process_group:
                                rank_to_append = (
                                    process_group[0] if process_group[1] == check_rank else process_group[1]
                                )
                                if rank_to_append not in ring_for_rank:
                                    stable_status = False
                                    rank_to_check_list.append(rank_to_append)
                                    ring_for_rank.append(rank_to_append)

                ring.append(ring_for_rank)
                ranks_in_ring.extend(ring_for_rank)

            return ring

        assert _power_of_two(self.world_size)
        power_of_two = int(math.log2(self.world_size))
        median = power_of_two // 2
        balanced_logical_mesh_shape = (2**median, 2 ** (power_of_two - median))
        row_size, column_size = balanced_logical_mesh_shape[0], balanced_logical_mesh_shape[1]
        balanced_logical_mesh = []
        for row_index in range(row_size):
            balanced_logical_mesh.append([])
            for column_index in range(column_size):
                balanced_logical_mesh[row_index].append(self.physical_devices[row_index * column_size + column_index])

        homogeneous_device_dict = _detect_homogeneous_device(self.alpha_beta_dict)
        beta_list = [b for b in homogeneous_device_dict.keys()]
        beta_list.sort()
        beta_list.reverse()
        homogeneous_types = len(beta_list)
        best_logical_mesh = None
        if homogeneous_types >= 2:
            for _ in range(homogeneous_types - 1):
                lowest_beta = beta_list.pop()
                best_homogeneous_group = homogeneous_device_dict[lowest_beta]
                # if the best homogeneous group contains all physical devices,
                # we will build the logical device mesh based on it. Otherwise,
                # we will check next level homogeneous group.
                if _check_contain_all_devices(best_homogeneous_group):
                    # We choose the largest ring for each rank to maximum the best bus utilization.
                    best_logical_mesh = _construct_largest_ring(best_homogeneous_group)
                    break

        if homogeneous_types == 1 or best_logical_mesh is None:
            # in this case, we use balanced logical mesh as the best
            # logical mesh.
            best_logical_mesh = balanced_logical_mesh

        return best_logical_mesh

    def extract_alpha_beta_for_device_mesh(self):
        """
        Extract the mesh_alpha list and mesh_beta list based on the
            best logical mesh, which will be used to initialize the device mesh.

        Usage:
            >>> physical_devices = [0, 1, 2, 3]
            >>> ab_profiler = AlphaBetaProfiler(physical_devices)
            >>> mesh_alpha, mesh_beta = profiler.extract_alpha_beta_for_device_mesh()
            >>> print(mesh_alpha)
            [2.5917552411556242e-05, 0.00010312341153621673]
            >>> print(mesh_beta)
            [5.875573704655635e-11, 4.7361584445959614e-12]
        """
        best_logical_mesh = self.search_best_logical_mesh()

        first_axis = [row[0] for row in best_logical_mesh]
        second_axis = best_logical_mesh[0]

        # init process group for both axes
        first_axis_process_group = dist.new_group(first_axis)
        second_axis_process_group = dist.new_group(second_axis)

        # extract alpha and beta for both axes
        def _extract_alpha_beta(pg, pg_handler):
            latency = self.profile_latency(pg, pg_handler)
            bandwidth = self.profile_bandwidth(pg, pg_handler)
            broadcast_object = [latency, bandwidth]
            dist.broadcast_object_list(broadcast_object, src=pg[0])
            return broadcast_object

        first_latency, first_bandwidth = _extract_alpha_beta(first_axis, first_axis_process_group)
        second_latency, second_bandwidth = _extract_alpha_beta(second_axis, second_axis_process_group)
        mesh_alpha = [first_latency, second_latency]
        # The beta values have been enlarged by 1e10 times temporarily because the computation cost
        # is still estimated in the unit of TFLOPs instead of time. We will remove this factor in future.
        mesh_beta = [1e10 / first_bandwidth, 1e10 / second_bandwidth]

        return mesh_alpha, mesh_beta
