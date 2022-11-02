import torch
import torch.nn as nn

from .msa import MSAStack
from .ops import OutProductMean
from .triangle import PairStack


class EvoformerBlock(nn.Module):

    def __init__(self, d_node, d_pair):
        super(EvoformerBlock, self).__init__()

        self.msa_stack = MSAStack(d_node, d_pair, p_drop=0.15)
        self.communication = OutProductMean(n_feat=d_node, n_feat_out=d_pair, n_feat_proj=32)
        self.pair_stack = PairStack(d_pair=d_pair)

    def forward(self, node, pair):
        node = node + self.msa_stack(node, pair)
        pair = pair + self.communication(node)
        pair = pair + self.pair_stack(pair)
        return node, pair


class Evoformer(nn.Module):

    def __init__(self, d_node, d_pair):
        super(Evoformer, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(3):
            self.blocks.append(EvoformerBlock(d_node, d_pair))

    def forward(self, node, pair):
        for b in self.blocks:
            node, pair = b(node, pair)
        return node, pair

def evoformer_base():
    return Evoformer(d_node=256, d_pair=128)


def evoformer_large():
    return Evoformer(d_node=512, d_pair=256)


__all__ = ['Evoformer', 'evoformer_base', 'evoformer_large']
