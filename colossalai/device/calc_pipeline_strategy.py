from math import pow

import numpy as np


def get_submesh_choices(num_hosts, num_devices_per_host, mode="new"):
    submesh_choices = []
    i = 1
    p = -1
    while i <= num_devices_per_host:
        i *= 2
        p += 1
    assert pow(2, p) == num_devices_per_host, ("Only supports the cases where num_devices_per_host is power of two, "
                                               f"while now num_devices_per_host = {num_devices_per_host}")
    if mode == "alpa":
        for i in range(p + 1):
            submesh_choices.append((1, pow(2, i)))
        for i in range(2, num_hosts + 1):
            submesh_choices.append((i, num_devices_per_host))
    elif mode == "new":
        for i in range(p // 2 + 1):
            for j in range(i, p - i + 1):
                submesh_choices.append((pow(2, i), pow(2, j)))
    return submesh_choices


def alpa_dp_impl(num_layers, num_devices, num_microbatches, submesh_choices, compute_cost, max_stage_cost,
                 best_configs):
    """Implementation of Alpa DP for pipeline strategy
	Paper reference: https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf

	Arguments:
		num_layers: K
		num_devices: N*M
		num_microbatches: B
		submesh_choices: List[(n_i,m_i)]
		compute_cost: t_intra
	"""
    # For f, layer ID start from 0
    # f[#pipeline stages, layer id that is currently being considered, number of devices used]
    f = np.full((num_layers + 1, num_layers + 1, num_devices + 1), np.inf, dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_layers + 1, num_devices + 1), 0.0, dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 3), -1, dtype=np.int32)
    f[0, num_layers, 0] = 0
    for s in range(1, num_layers + 1):
        for k in range(num_layers - 1, -1, -1):
            for d in range(1, num_devices + 1):
                for m, submesh in enumerate(submesh_choices):
                    n_submesh_devices = np.prod(np.array(submesh))
                    if n_submesh_devices <= d:
                        # TODO: [luzgh]: Why alpa needs max_n_succ_stages? Delete.
                        # if s - 1 <= max_n_succ_stages[i, k - 1, m, n_config]:
                        # ...
                        for i in range(num_layers, k, -1):
                            stage_cost = compute_cost[k, i, m]
                            new_cost = f[s - 1, k, d - n_submesh_devices] + stage_cost
                            if (stage_cost <= max_stage_cost and new_cost < f[s, k, d]):
                                f[s, k, d] = new_cost
                                f_stage_max[s, k, d] = max(stage_cost, f_stage_max[s - 1, i, d - n_submesh_devices])
                                f_argmin[s, k, d] = (i, m, best_configs[k, i, m])
    best_s = -1
    best_total_cost = np.inf
    for s in range(1, num_layers + 1):
        if f[s, 0, num_devices] < best_total_cost:
            best_s = s
            best_total_cost = f[s, 0, num_devices]

    if np.isinf(best_total_cost):
        return np.inf, None

    total_cost = f[best_s, 0, num_devices] + (num_microbatches - 1) * f_stage_max[best_s, 0, num_devices]
    current_s = best_s
    current_layer = 0
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer < num_layers and current_devices > 0:
        next_start_layer, submesh_choice, autosharding_choice = (f_argmin[current_s, current_layer, current_devices])
        assert next_start_layer != -1 and current_devices != -1
        res.append(((current_layer, next_start_layer), submesh_choice, autosharding_choice))
        current_s -= 1
        current_layer = next_start_layer
        current_devices -= np.prod(np.array(submesh_choices[submesh_choice]))
    assert (current_s == 0 and current_layer == num_layers and current_devices == 0)

    return total_cost, res


def alpa_dp(num_layers,
            num_devices,
            num_microbatches,
            submesh_choices,
            num_autosharding_configs,
            compute_cost,
            gap=1e-6):
    """Alpa auto stage dynamic programming.
	Code reference: https://github.com/alpa-projects/alpa/blob/main/alpa/pipeline_parallel/stage_construction.py

    Arguments:
        submesh_choices: List[(int,int)]
        num_autosharding_configs: Max number of t_intra(start_layer, end_layer, LogicalMesh)
        compute_cost: np.array(num_layers,num_layers,num_submesh_choices,num_autosharding_configs)
    """
    assert np.shape(compute_cost) == (num_layers, num_layers, len(submesh_choices),
                                      num_autosharding_configs), "Cost shape wrong."
    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # TODO: [luzgh]: Why alpa needs the num_autosharding_configs dimension in compute_cost?
    # In dp_impl it seems the argmin n_config will be chosen. Just amin here.
    best_configs = np.argmin(compute_cost, axis=3)
    best_compute_cost = np.amin(compute_cost, axis=3)
    assert len(all_possible_stage_costs), "no solution in auto stage construction."
    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = alpa_dp_impl(num_layers, num_devices, num_microbatches, submesh_choices, best_compute_cost,
                                      max_stage_cost, best_configs)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_cost

    return best_cost, best_solution
