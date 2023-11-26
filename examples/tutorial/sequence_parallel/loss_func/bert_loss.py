import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc


class BertLoss(nn.Module):
    def forward(self, lm_loss, sop_logits, loss_mask, sentence_order):
        lm_loss_ = lm_loss.float()
        loss_mask = loss_mask.float()
        loss_mask_sum = loss_mask.sum()
        lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1))

        lm_loss /= loss_mask_sum

        torch.distributed.all_reduce(lm_loss, group=gpc.get_group(ParallelMode.SEQUENCE))

        if sop_logits is not None:
            sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
            sop_loss = sop_loss.float()
            loss = lm_loss + sop_loss * gpc.get_world_size(ParallelMode.SEQUENCE)
        else:
            sop_loss = None
            loss = lm_loss

        return loss
