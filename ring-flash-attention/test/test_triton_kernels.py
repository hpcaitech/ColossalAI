import torch
from ring_flash_attn.triton_utils import flatten_varlen_lse as triton_flatten_varlen_lse
from ring_flash_attn.triton_utils import unflatten_varlen_lse as triton_unflatten_varlen_lse
from ring_flash_attn.utils import flatten_varlen_lse, unflatten_varlen_lse

if __name__ == "__main__":
    device = torch.device("cuda:0")

    cu_seqlens = [0, 15, 156, 529]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    batch_size = len(cu_seqlens) - 1
    max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
    n_head = 5

    lse = torch.randn((batch_size, n_head, max_seqlen), dtype=torch.float32, device=device)
    flatten_lse = flatten_varlen_lse(lse, cu_seqlens_tensor)
    triton_flatten_lse = triton_flatten_varlen_lse(lse, cu_seqlens_tensor)
    assert torch.all(flatten_lse == triton_flatten_lse)

    flatten_lse = flatten_lse.transpose(-2, -1).unsqueeze(dim=-1)
    triton_flatten_lse = triton_flatten_lse.transpose(-2, -1).unsqueeze(dim=-1)

    unflatten_lse = unflatten_varlen_lse(flatten_lse, cu_seqlens_tensor, max_seqlen)
    triton_unflatten_lse = triton_unflatten_varlen_lse(triton_flatten_lse, cu_seqlens_tensor, max_seqlen)

    for i in range(batch_size):
        seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
        assert torch.all(
            unflatten_lse[i, :, :seqlen] == triton_unflatten_lse[i, :, :seqlen]
        ), f"{unflatten_lse[i, :seqlen]} vs {triton_unflatten_lse[i, :seqlen]}"
