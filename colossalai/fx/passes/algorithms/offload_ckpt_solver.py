from typing import List, Tuple
import torch
from torch.fx import GraphModule, Node
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.profiler import parameter_size
import math
from .linearize import linearize
from .utils import *
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.codegen.activation_checkpoint_codegen import _find_nested_ckpt_regions
from colossalai.fx.passes.algorithms.ckpt_solver_rotor import _construct_chain, _compute_table

INF = float("inf")


class pofo_table:

    def __init__(self, chain_length, nb_slots):
        # opt[i][mf][(df, db)]
        self.L = chain_length
        self.nb_slots = nb_slots
        self.opt = {
            False: [[{} for _ in range(self.nb_slots + 1)] for _ in range(chain_length + 1)],
            True: [[{} for _ in range(self.nb_slots + 1)] for _ in range(chain_length + 1)]
        }
        self.what = {
            False: [[{} for _ in range(self.nb_slots + 1)] for _ in range(chain_length + 1)],
            True: [[{} for _ in range(self.nb_slots + 1)] for _ in range(chain_length + 1)]
        }

        # what[has_bar][i][mf][(df, db)] = (is_enable, is_offload, index)
        #     where index is only present if not is_enable

    def get_value(self, state, table, default):
        i, A, df, db, has_bar = state
        if A + df > self.nb_slots or A + db > self.nb_slots:
            return default
        try:
            res = table[has_bar][i][A][(df, db)]
        except KeyError as e:
            print("GV Not Found", has_bar, i, A, df, db)
            raise e
        return res

    def get_opt(self, state):
        return self.get_value(state, self.opt, INF)

    def get_what(self, state):
        return self.get_value(state, self.what, None)

    def set_value(self, state, opt, what):
        i, A, df, db, has_bar = state
        self.opt[has_bar][i][A][(df, db)] = opt
        self.what[has_bar][i][A][(df, db)] = what

    def _construct_chain_pofo(node_list, data, mem_unit):
        pass


def solver_pofo(gm: ColoGraphModule,
                data,
                band_width: int,
                mem_limit: int,
                mem_slots: int = 500,
                cnode: List[str] = None,
                eps: float = 0.0) -> ColoGraphModule:
    """Solver that combine offload and activation checkpoint
    Reference: https://proceedings.neurips.cc/paper/2021/hash/c8461bf13fca8a2b9912ab2eb1668e4b-Abstract.html

    Args:
        gm (ColoGraphModule): ColoGraphModule derived from tracer
        data (_type_): input of the model
        band_width (int): offload bandwidth, unit Byte
        mem_limit (int): memory limit, unit Byte
        mem_slots (int, optional): number of memory slots. Defaults to 500.
        cnode (List[str], optional): common node for linearize. Defaults to None.
        eps (float, optional): epsilon for memory decay. Defaults to 0.02.

    Returns:
        ColoGraphModule: annotated graph module
    """

    node_list = linearize(gm, cnode)
    mem_limit -= parameter_size(gm)
    mem_unit = mem_limit * (1 - eps) // mem_slots
    MetaInfoProp(gm).run(data)
    chain: Chain = _construct_chain_pofo(node_list, data, mem_unit)
    opt, what = _compute_table(chain, mem_slots)
