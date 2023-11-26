from typing import Any, Dict, List, Tuple

import torch
from torch.fx import Graph, Node

from .region import Region
from .solver import SolverFactory
from .training_simulator import TrainingSimulator
from .util import NodeInfo


class RegionManager:
    """
    RegionManager is used to construct and manage the offload plan for the model execution.

    Args:
        graph (Graph): a Graph object used for analysis and strategy generation.
        solver_name (str): a solver name which specifies the preferences for plan searching.
        memory_budget (float): the given memory budget.
        cnode (List[str], optional): Common node List, should be the subset of input.
    """

    def __init__(self, graph: Graph, solver_name: str = "asyn", memory_budget: float = -1.0, cnode: List[str] = None):
        self.graph = graph
        assert graph.owning_module is not None, "The given graph is not associated with a owning_module"
        self.root_module = self.graph.owning_module
        self.nodes = list(graph.nodes)
        self.cnode = cnode
        self.only_param_ops = []
        self.param_region_map: Dict[torch.nn.Parameter, Region] = dict()
        self.shared_region_pairs: List[Tuple[Region, Region]] = list()
        self.region_list: List[Region] = list()
        self.rid_in_pool: List[int] = list()
        self.mem_block_size: int = 0
        self.memory_budget = memory_budget

        self.solver_name = solver_name
        self.require_pool: bool = solver_name == "asyn"

        self.reg_to_block: Dict[int, int] = dict()

    def _build_regions(self):
        """
        1. Pre-processing, mainly contains linearized computing graph and
            merge smaller regions into larger ones.
        2. Construct a solver to search for an efficient offload strategy.
        3. Post-processing, mainly contains early region placement if using asynchronous mode,
            and initialize region data.
        """

        self._pre_process()

        solver_cls = SolverFactory.create(self.solver_name)
        solver = solver_cls(self.region_list, self.memory_budget)
        solver._call_solver()

        self._post_process(solver.best_ts)

    def _pre_process(self):
        init_region_list = self._linearize_graph()

        if len(self.shared_region_pairs) > 1:
            raise NotImplementedError("The current version only considers at most one pair of parameter sharing.")

        elif len(self.shared_region_pairs) == 1:
            shared_regs = self.shared_region_pairs[0]
            assert shared_regs[0].shared_rid == shared_regs[1].r_id and shared_regs[1].shared_rid == shared_regs[0].r_id
            fst_id = shared_regs[0].r_id
            lst_id = shared_regs[1].r_id
            regs_left_out = init_region_list[: fst_id + 1]
            regs_right_out = init_region_list[lst_id:]
            hold_regs = init_region_list[fst_id + 1 : lst_id]
        else:
            regs_left_out = []
            regs_right_out = []
            hold_regs = init_region_list

        self.mem_block_size = self._search_block_size(hold_regs)
        hold_regs = self._merge_small_regions(hold_regs)

        if self.require_pool:
            for reg in hold_regs:
                reg.in_mem_pool_flag = True
                self.rid_in_pool.append(reg.r_id)

        self.region_list.extend(regs_left_out)
        self.region_list.extend(hold_regs)

        for reg in regs_right_out:
            reg.r_id = self.region_list[-1].r_id + 1
            self.region_list[reg.shared_rid].shared_rid = reg.r_id
            self.region_list.append(reg)

        self._process_shared_region()

        self.max_param_num = max([reg.param_num for reg in self.region_list])
        self.memory_budget -= self.max_param_num * torch.tensor([], dtype=torch.float32).element_size()

    def _post_process(self, ts: TrainingSimulator = None):
        if self.require_pool:
            self._early_region_placement(ts)
        self._init_region_data()

    def _early_region_placement(self, ts: TrainingSimulator):
        """
        Implemented the early region placement strategy to avoid GPU memory fragmentation.
        It maps all region data into a contiguous memory space and
        reuses the same memory space for regions that do not coexist.

        Args:
            ts (TrainingSimulator): the best training simulator, which records region execution flow.

        Raises:
            NotImplementedError: due to the naive implementation,
                it may not find a suitable region placement strategy for the given execution flow.
        """

        reg_flow = torch.cat([ts.fwd_reg_flow, ts.bwd_reg_flow], dim=0)
        mem_block_num = torch.max(torch.sum(reg_flow[:, self.rid_in_pool], dim=1))
        coexist_matrix = torch.logical_or(ts.fwd_reg_flow, ts.bwd_reg_flow)

        block_to_regs = {}
        for block_idx in range(mem_block_num):
            block_to_regs[block_idx] = []
        for reg in self.region_list:
            if reg.r_id in self.rid_in_pool:
                cur_reg_appears = coexist_matrix[:, reg.r_id]
                cur_reg_coexists = torch.sum(coexist_matrix[cur_reg_appears], dim=0).bool()
                for block_idx in range(mem_block_num):
                    if not any(cur_reg_coexists[block_to_regs[block_idx]]):
                        block_to_regs[block_idx].append(reg.r_id)
                        self.reg_to_block[reg.r_id] = block_idx
                        break

                if reg.r_id not in self.reg_to_block:
                    raise NotImplementedError(
                        f"can not find a block from the memory pool to store parameters of the region"
                    )
        self.memory_pool = torch.chunk(
            torch.zeros(int(mem_block_num * self.mem_block_size / 2), dtype=torch.half, device="cuda"),
            chunks=int(mem_block_num),
        )

    def _merge_small_regions(self, orig_reg_list: List[Region]) -> List[Region]:
        """
        Merge smaller regions into larger ones for better bandwidth utilization and easier management.
        It is inspired by Gemini.

        Args:
            orig_reg_list (List[Region]): original region list.

        Returns:
            List[Region]: region list after merging.
        """

        r_id = orig_reg_list[0].r_id
        region = Region(r_id=r_id)
        region_list = [region]

        for orig_reg in orig_reg_list:
            if region_list[-1].param_size + orig_reg.param_size > self.mem_block_size:
                r_id += 1
                region = Region(r_id=r_id)
                region_list.append(region)
            region.param_size += orig_reg.param_size
            region.param_num += orig_reg.param_num
            region.nodes.extend(orig_reg.nodes)
            region.fp16_params.extend(orig_reg.fp16_params)
            self.__update_param_region_map(orig_reg.fp16_params, region)

        return region_list

    def _search_block_size(
        self, region_list: List[Region], search_interval_byte: int = 1024, search_range_byte: int = 128 * 1024**2
    ) -> int:
        """
        Search for a suitable memory block size.

        Args:
            region_list (List[Region]): region list.
            search_interval_byte (int): searching interval in byte.
            search_range_byte (int): searching range in byte.

        Returns:
            int: the best memory block size.
        """

        def _get_wasted_mem(size_list: List[int], blk_size: int):
            """
            Get wasted byte for a certain block size.
            """
            acc_wasted = 0
            left = 0
            for s in size_list:
                if left + s > blk_size:
                    acc_wasted += blk_size - left
                    left = s
                left += s
            acc_wasted += blk_size - left
            return acc_wasted

        param_size_list = [region.param_size for region in region_list if region.r_id == region.shared_rid]

        start_size = max(param_size_list)
        min_mem_waste = float("+inf")
        best_block_size = start_size

        for block_size in range(start_size, start_size + search_range_byte + 1, search_interval_byte):
            temp_waste = 0
            temp_waste += _get_wasted_mem(param_size_list, block_size)
            if temp_waste < min_mem_waste:
                min_mem_waste = temp_waste
                best_block_size = block_size

        return best_block_size

    def _init_region_data(self):
        """
        Initialize region data, which maps the parameters in the region to a contiguous memory space.
        """

        self.temp_fp32_data = torch.zeros(self.max_param_num, device="cuda", dtype=torch.float32)

        for region in self.region_list:
            pre_alloc_tensor = None
            if self.require_pool and region.r_id in self.rid_in_pool:
                block_idx = self.reg_to_block[region.r_id]
                pre_alloc_tensor = self.memory_pool[block_idx]

            if region.r_id <= region.shared_rid:
                region.init_param_data(pre_alloc_tensor)
            else:
                shared_region = self.region_list[region.shared_rid]
                region.fp16_data = shared_region.fp16_data
                region.fp32_data = shared_region.fp32_data
                region.param_to_range = shared_region.param_to_range
            region.temp_fp32_data = self.temp_fp32_data[: region.param_num].detach()

        torch.cuda.empty_cache()

    def _process_shared_region(self):
        """
        Special processing for the shared region, which uses GPT2 and Bert case as a priori knowledge.
        """

        if len(self.shared_region_pairs):
            assert len(self.shared_region_pairs) <= 1
            former_reg, latter_reg = self.shared_region_pairs[0]
            assert latter_reg.param_num >= former_reg.param_num
            embedding_node = former_reg.nodes[-1]
            assert embedding_node.op == "call_module" and isinstance(
                self.root_module.get_submodule(embedding_node.target), torch.nn.Embedding
            )
            if latter_reg.param_num > former_reg.param_num:
                for idx, n in enumerate(latter_reg.nodes):
                    if (
                        n.op == "call_module" and isinstance(self.root_module.get_submodule(n.target), torch.nn.Linear)
                    ) or (n.op == "call_function" and n.target is torch.nn.functional.linear):
                        cut_node_idx = idx + 1
                        break
                assert len(latter_reg.fp16_params) == 2
                new_reg = latter_reg.split(cut_node_idx, 1)
                for p in new_reg.fp16_params:
                    self.param_region_map[p] = new_reg
                self.region_list.insert(new_reg.r_id, new_reg)
                for reg in self.region_list[new_reg.r_id + 1 :]:
                    reg.r_id += 1
            latter_reg.shared_rid = former_reg.r_id
            former_reg.shared_rid = latter_reg.r_id

    def _linearize_graph(self) -> List[Region]:
        """Linearizing the graph

        Args:
            graph (Graph): The computing graph to be optimized.

        Returns:
            List[Region]: each region contains the actual 'node' in linearized manner.

        Remarks:
            Do merge the inplace ops and shape-consistency ops into the previous node.
        """

        # List of target name that could be seen as common node
        common_ops = ["getattr", "getitem", "size"]

        def _is_cop(target: Any) -> bool:
            """Check if an op could be seen as common node

            Args:
                target (Any): node target

            Returns:
                bool
            """

            if isinstance(target, str):
                return target in common_ops
            else:
                return target.__name__ in common_ops

        def _is_act(data: Any) -> bool:
            """Check if an op could be seen as parameter computation start

            Args:
                data (Any): meta_data

            Returns:
                bool
            """

            label = False
            if isinstance(data, torch.Tensor):
                return True
            elif isinstance(data, (tuple, list)):
                for d in data:
                    label = label or _is_act(d)
            return label

        def _maybe_param_comp_start() -> bool:
            """Check if an op could be seen as parameter computation start

            Args:
                n (Node): node

            Returns:
                bool
            """

            label = False
            if n.op == "get_attr":
                label = True
            elif n.op == "call_module":
                target = n.target
                submod = self.root_module.get_submodule(target)
                if (
                    len(list(submod.named_parameters(recurse=False))) != 0
                    or len(list(submod.named_buffers(recurse=False))) != 0
                ):
                    label = True

            return label and not sum([v for _, v in param_op_deps.items()])

        def _is_param_comp_end() -> bool:
            """Check if an op could be seen as parameter computation end

            Args:
                n (Node): node

            Returns:
                bool
            """

            def _is_inplace(n: Node):
                """Get the inplace argument from ``torch.fx.Node``"""
                inplace = False
                if n.op == "call_function":
                    inplace = n.kwargs.get("inplace", False)
                elif n.op == "call_module":
                    inplace = getattr(n.graph.owning_module.get_submodule(n.target), "inplace", False)
                return inplace

            label = False

            if n.op == "call_module":
                target = n.target
                submod = self.root_module.get_submodule(target)
                if (
                    len(list(submod.named_parameters(recurse=False))) != 0
                    or len(list(submod.named_buffers(recurse=False))) != 0
                ):
                    label = True

            elif n.op == "call_function":
                label = any(map(lambda x: x.name in self.only_param_ops, n.all_input_nodes)) and any(
                    map(lambda x: x.name not in self.only_param_ops and not _is_cop(n.target), n.all_input_nodes)
                )

            return label and not sum([v for _, v in param_op_deps.items()]) and not any(map(_is_inplace, n.users))

        def _exception_node_handling():
            # TODO meta info prop bug
            if n.name.__contains__("transpose") and n.meta["fwd_out"][0].dim() <= 2:
                n.meta["fwd_out"] = []

        # make sure that item in cnode is valid
        if self.cnode:
            for name in self.cnode:
                try:
                    assert (
                        next(node for node in self.graph.nodes if node.name == name).op == "placeholder"
                    ), f"Common node {name} is not an input of the model."
                except StopIteration:
                    raise ValueError(f"Common node name {name} not in graph.")
        else:
            self.cnode = []

        node_id = 0
        region_id = 0

        param_op_deps = {}

        deps = {}
        region_list = []
        region = Region(r_id=region_id)

        act_n = None

        for n in self.graph.nodes:
            if n.op != "placeholder" and n.op != "output":
                for n_par in n.all_input_nodes:
                    if n_par.op != "placeholder" and n_par.name not in self.cnode:
                        deps[n_par] -= 1
                    if n_par.op != "placeholder" and n_par.name in self.only_param_ops:
                        param_op_deps[n_par] -= 1

                if act_n in region.nodes and _maybe_param_comp_start():
                    ns = []
                    border_n_idx = region.nodes.index(act_n)
                    if border_n_idx < len(region.nodes):
                        ns = region.nodes[border_n_idx + 1 :]
                        region.nodes = region.nodes[: border_n_idx + 1]
                    region_list.append(region)
                    region_id += 1
                    region = Region(r_id=region_id)
                    region.nodes = ns

                _exception_node_handling()
                region.nodes.append(n)
                self._set_node_and_region_info(node_id, n, region)
                node_id += 1

                # if the node could free all dependencies in graph
                # we could begin a new region
                if _is_param_comp_end():
                    region_list.append(region)
                    region_id += 1
                    region = Region(r_id=region_id)

                # propagate common node attr if possible
                if len(n.all_input_nodes) == len(
                    [node for node in n.all_input_nodes if node.name in self.cnode]
                ) or _is_cop(n.target):
                    self.cnode.append(n.name)
                else:
                    deps[n] = len([user for user in n.users if user.op != "output"])

                # propagate param node attr if possible
                if (
                    len(n.all_input_nodes)
                    == len([node for node in n.all_input_nodes if node.name in self.only_param_ops])
                    or n.op == "get_attr"
                ):
                    self.only_param_ops.append(n.name)
                    param_op_deps[n] = len([user for user in n.users if user.op != "output"])

                # record last activation node
                if _is_act(n._meta_data):
                    act_n = n

        if len(region.nodes):
            region_list.append(region)

        return region_list

    def _set_node_and_region_info(self, node_id: int, cur_n: Node, cur_reg: Region):
        cur_n.node_info = NodeInfo(node_id)

        if cur_n.op == "call_module":
            target = cur_n.target
            submod = self.root_module.get_submodule(target)
            for p in list(submod.parameters(recurse=False)):
                if p in self.param_region_map:
                    cur_reg.shared_rid = self.param_region_map[p].r_id
                    self.param_region_map[p].shared_rid = cur_reg.r_id
                    self.shared_region_pairs.append((self.param_region_map[p], cur_reg))
                else:
                    self.param_region_map[p] = cur_reg

                cur_reg.fp16_params.append(p)
                cur_reg.param_num += p.data.numel()
                cur_reg.param_size += p.data.numel() * p.data.element_size()

        elif cur_n.op == "get_attr":
            attr_itr = self.root_module
            atoms = cur_n.target.split(".")
            for atom in atoms:
                attr_itr = getattr(attr_itr, atom)

            if isinstance(attr_itr, torch.nn.Parameter):
                if attr_itr in self.param_region_map:
                    cur_reg.shared_rid = self.param_region_map[attr_itr].r_id
                    self.param_region_map[attr_itr].shared_rid = cur_reg.r_id
                    self.shared_region_pairs.append((self.param_region_map[attr_itr], cur_reg))
                else:
                    self.param_region_map[attr_itr] = cur_reg

                cur_reg.fp16_params.append(attr_itr)
                cur_reg.param_num += attr_itr.data.numel()
                cur_reg.param_size += attr_itr.data.numel() * attr_itr.data.element_size()

    def get_region(self, param: torch.nn.Parameter) -> Region:
        """
        Return the region owning the parameter.

        Args:
            param (torch.nn.Parameter): a torch parameter object
        """
        return self.param_region_map[param]

    def __update_param_region_map(self, params: List[torch.nn.Parameter], region: Region):
        for p in params:
            self.param_region_map[p] = region
