import numpy as np


def alpa_dp_impl(num_layers, num_devices, num_microbatches, submesh_choices, num_autosharding_configs, compute_cost,
                 max_stage_cost):
    """Implementation of Alpa DP for pipeline strategy
	Paper reference: https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf

	Arguments:
		num_layers: K
		num_devices: N*M
		num_microbatches: B
		submesh_choices: List[(n_i,m_i)]
		num_autosharding_configs: Max number of t_intra(start_layer, end_layer, LogicalMesh)
		compute_cost: t_intra
	"""

    pass


def alpa_dp(num_layers,
            num_devices,
            num_microbatches,
            submesh_choices,
            num_autosharding_configs,
            compute_cost,
            gap=1e-6):
    """Alpa auto stage dynamic programming.
	Code reference: https://github.com/alpa-projects/alpa

    Arguments:
        submesh_choices: List[(int,int)]
        compute_cost: np.array(num_layers,num_layers,num_submesh_choices,num_autosharding_configs)
    """
    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    assert len(all_possible_stage_costs), "no solution in auto stage construction."
    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = alpa_dp_impl(num_layers, num_devices, num_microbatches, submesh_choices,
                                      num_autosharding_configs, compute_cost, max_stage_cost)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_cost

    return best_cost, best_solution
