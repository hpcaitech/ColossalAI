import torch


class GPTQManager:
    def __init__(self, quant_config, max_input_len: int = 1):
        self.max_dq_buffer_size = 1
        self.max_inner_outer_dim = 1
        self.bits = quant_config.bits
        self.use_act_order = quant_config.desc_act
        self.max_input_len = 1
        self.gptq_temp_state_buffer = None
        self.gptq_temp_dq_buffer = None
        self.quant_config = quant_config

    def post_init_gptq_buffer(self, model: torch.nn.Module) -> None:
        from .cai_gptq import CaiQuantLinear

        HAS_GPTQ_CUDA = False
        try:
            from colossalai.kernel.op_builder.gptq import GPTQBuilder

            gptq_cuda = GPTQBuilder().load()
            HAS_GPTQ_CUDA = True
        except ImportError:
            warnings.warn("CUDA gptq is not installed")
            HAS_GPTQ_CUDA = False

        for name, submodule in model.named_modules():
            if isinstance(submodule, CaiQuantLinear):
                self.max_dq_buffer_size = max(self.max_dq_buffer_size, submodule.qweight.numel() * 8)

                if self.use_act_order:
                    self.max_inner_outer_dim = max(
                        self.max_inner_outer_dim, submodule.infeatures, submodule.outfeatures
                    )
                self.bits = submodule.bits
        if not (HAS_GPTQ_CUDA and self.bits == 4):
            return

        max_input_len = 1
        if self.use_act_order:
            max_input_len = self.max_input_len
        # The temp_state buffer is required to reorder X in the act-order case.
        # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
        self.gptq_temp_state_buffer = torch.zeros(
            (max_input_len, self.max_inner_outer_dim), dtype=torch.float16, device=torch.cuda.current_device()
        )
        self.gptq_temp_dq_buffer = torch.zeros(
            (1, self.max_dq_buffer_size), dtype=torch.float16, device=torch.cuda.current_device()
        )

        gptq_cuda.prepare_buffers(
            torch.device(torch.cuda.current_device()), self.gptq_temp_state_buffer, self.gptq_temp_dq_buffer
        )
        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        gptq_cuda.set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

        torch.cuda.empty_cache()
