import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

if HAS_TRITON:
    """
    softmax kernel is modified based on
    https://github.com/openai/triton/blob/34817ecc954a6f4ca7b4dfb352fdde1f8bd49ca5/python/tutorials/02-fused-softmax.py
    """

    @triton.jit
    def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
        r"""the kernel function for implementing softmax operator
        Args:
            output_ptr: the output after finishing softmax operation, (N, hidden_dim)
            input_ptr: the tensor of input, shape should be (N, hidden_dim)
            n_cols(tl.constexpr): the number of cols of input
            BLOCK_SIZE(tl.constexpr): the block_size of your hidden_dim dimension, typically BLOCK_SIZE >= hidden_dim
        """
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
        row_minus_max = row - tl.max(row, axis=0)

        if mask_ptr is not None:
            # load mask into SRAM
            mask_ptrs = (mask_ptr + (row_indx * row_stride)) + col_offsets
            mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=0).to(tl.float32)

            # update
            row_minus_max = row_minus_max + mask

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        # Write back output to DRAM
        tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

    def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
        if mask is not None:
            assert input[-1] == mask[-1], "the last dimentions should be the same for input and mask"
        assert dim == -1 or dim == len(input.shape) - 1, "currently softmax layer only support last dimention"

        hidden_dim = input.shape[-1]
        output = torch.empty_like(input)
        input = input.view(-1, hidden_dim)
        if mask is not None:
            mask = mask.view(-1, hidden_dim)
            assert input.shape[0] == mask.shape[0], "the fist dimention of mask and input should be the same"

        num_rows, num_cols = input.shape
        block_size = max(triton.next_power_of_2(num_cols), 2)
        num_warps = 16
        if block_size >= 4096:
            num_warps = 16
        elif block_size >= 2048:
            num_warps = 8
        else:
            num_warps = 4

        if num_rows <= 350000:
            grid = (num_rows,)
            softmax_kernel[grid](
                output, input, input.stride(0), num_cols, mask, BLOCK_SIZE=block_size, num_warps=num_warps
            )
        else:
            grid = lambda meta: ()

            grid = lambda meta: (triton.cdiv(num_rows, meta["BLOCK_M"]),)

            if block_size >= 4096:
                pass
            elif block_size >= 2048:
                pass

            softmax_kernel[grid](
                output_ptr=output,
                input_ptr=input,
                row_stride=input.stride(0),
                n_rows=num_rows,
                n_cols=num_cols,
                mask_ptr=mask,
                # currently manually setting up size
                BLOCK_M=32,
                BLOCK_SIZE=block_size,
            )

        return output
