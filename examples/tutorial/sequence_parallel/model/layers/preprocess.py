import torch
import torch.nn as nn

from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc


class PreProcessor(nn.Module):
    def __init__(self, sub_seq_length):
        super().__init__()
        self.sub_seq_length = sub_seq_length

    def bert_position_ids(self, token_ids):
        # Create position ids
        seq_length = token_ids.size(1)
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        position_ids = torch.arange(
            seq_length * local_rank, seq_length * (local_rank + 1), dtype=torch.long, device=token_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        return position_ids

    def bert_extended_attention_mask(self, attention_mask):
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        start_index = local_rank * self.sub_seq_length
        end_index = (local_rank + 1) * self.sub_seq_length

        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)
        # [b, s, 1]
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        # [b, s/D, s]
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1

        attention_mask_bss = attention_mask_bss[:, start_index:end_index, :]

        # [b, 1, s/D, s]
        extended_attention_mask = attention_mask_bss.unsqueeze(1)

        # Convert attention mask to binary:
        extended_attention_mask = extended_attention_mask < 0.5

        return extended_attention_mask

    def forward(self, input_ids=None, attention_mask=None):
        if attention_mask is not None:
            extended_attention_mask = self.bert_extended_attention_mask(attention_mask)
        else:
            extended_attention_mask = None

        if input_ids is not None:
            position_ids = self.bert_position_ids(input_ids)
        else:
            position_ids = None
        return position_ids, extended_attention_mask
