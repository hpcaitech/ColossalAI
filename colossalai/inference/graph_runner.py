from typing import Dict, List

import torch
from torch import nn

from colossalai.inference.config import InputMetaData
from colossalai.logging import get_dist_logger


class CUDAGraphRunner:
    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}
        self.logger = get_dist_logger(__name__)

    def capture(
        self,
        input_tokens_ids: torch.Tensor,
        output_tensor: torch.Tensor,
        inputmetadata: InputMetaData,
        k_caches: List[torch.Tensor] = None,
        v_caches: List[torch.Tensor] = None,
        memory_pool=None,
    ) -> None:
        assert self.graph is None

        # run kernel once to cache the kernel, avoid stream capture error
        hidden_states_origin_model = self.model(
            input_tokens_ids,
            output_tensor,
            inputmetadata,
            k_caches,
            v_caches,
        )
        torch.cuda.synchronize()

        # Capture the graph.
        # self.logger.info(f"begin capture model...")
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):
            hidden_states_cuda_graph = self.model(
                input_tokens_ids,
                output_tensor,
                inputmetadata,
                k_caches,
                v_caches,
            )
        torch.cuda.synchronize()

        # Save the input and output buffers, because replay always uses the same virtual memory space
        self.input_buffers = {
            "input_tokens_ids": input_tokens_ids,
            "output_tensor": output_tensor,
            "block_tables": inputmetadata.block_tables,
            "sequence_lengths": inputmetadata.sequence_lengths,
            # "fd_inter_tensor_mid_output": inputmetadata.fd_inter_tensor._mid_output,
            # "fd_inter_tensor_mid_output_lse": inputmetadata.fd_inter_tensor._mid_output_lse,
            "k_caches": k_caches,
            "v_caches": v_caches,
        }
        self.output_buffers = {"logits": hidden_states_cuda_graph}
        return

    def forward(
        self,
        input_tokens_ids: torch.Tensor,
        output_tensor: torch.Tensor,
        inputmetadata: InputMetaData,
        k_caches: List[torch.Tensor] = None,
        v_caches: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Copy the input tensors to the input buffers.
        self.input_buffers["input_tokens_ids"].copy_(input_tokens_ids, non_blocking=True)
        self.input_buffers["output_tensor"].copy_(output_tensor, non_blocking=True)

        # for flexible block_table
        self.input_buffers["block_tables"].fill_(-1)
        M, N = inputmetadata.block_tables.shape
        self.input_buffers["block_tables"][:M, :N].copy_(inputmetadata.block_tables, non_blocking=True)

        self.input_buffers["sequence_lengths"].copy_(inputmetadata.sequence_lengths, non_blocking=True)

        # we only have a global fd_inter_tensor so we don't need to copy them
        # self.input_buffers["fd_inter_tensor_mid_output"].copy_(inputmetadata.fd_inter_tensor.mid_output, non_blocking=True)
        # self.input_buffers["fd_inter_tensor_mid_output_lse"].copy_(inputmetadata.fd_inter_tensor.mid_output_lse, non_blocking=True)

        # KV caches are fixed tensors, so we don't need to copy them.
        # self.input_buffers["k_caches"].copy_(k_caches, non_blocking=True)
        # self.input_buffers["v_caches"].copy_(v_caches, non_blocking=True)

        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["logits"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
