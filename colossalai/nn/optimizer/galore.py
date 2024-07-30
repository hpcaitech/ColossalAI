""" adapted from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/adamw8bit.py"""

import warnings
from typing import List

import torch
from bitsandbytes.optim.optimizer import Optimizer2State
from torch._C import _LinAlgError


def get_galore_param_groups(
    model, weight_decay, rank=256, update_proj_gap=200, scale=0.25, proj_type="std"
) -> List[dict]:
    """
    It's advised to use this instead of manually specifying which param groups
    to apply GaLore on.
    """
    galore_params = []
    non_galore = []
    no_decay_params = []
    no_decay = ["bias", "LayerNorm.weight"]

    for name, param in model.named_parameters():
        # Only make sense to do SVD on 2d gradient matrices
        # e.g. nn.Linear, VocabEmbedding, etc.
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        elif param.dim() == 2:
            galore_params.append(param)
        else:
            non_galore.append(param)

    param_groups = [
        {
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
            "weight_decay": weight_decay,
        },
        {"params": non_galore, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return param_groups


def make_low_rank_buffer(p, grad):
    """For compatibility with bitsandbytes's update_step, we need an empty low-rank
    param update buffer to avoid mutating original params.
    TODO: optimize by reusing the memory for p.grad? Need to modify bitsandbytes?
    """
    p.saved_data = p.data.clone()
    # p.data = grad.clone().to(p.data.dtype).to(p.data.device)
    p.data = torch.zeros_like(grad, device=grad.device, dtype=grad.dtype)
    # p.data.zero_()
    p.grad = grad


class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type="std"):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.svd_type = None

    def project(self, full_rank_grad, iter):
        dim = full_rank_grad.dim()
        if dim != 2:
            warnings.warn(
                f"Warning: You shouldn't specify projection rank for {dim}D params in param_groups. Skipping SVD."
            )
            return full_rank_grad

        m, n = full_rank_grad.shape  # For ZeRO sharded grads
        if self.proj_type == "std":
            # Project the lower dim to minimize information loss
            if self.svd_type is None:
                self.svd_type = "right" if m >= n else "left"
            # SVD step
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type=self.svd_type)
            if self.svd_type == "right":
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t()[:n])
            else:
                low_rank_grad = torch.matmul(self.ortho_matrix.t()[:, :m], full_rank_grad)

        elif self.proj_type == "reverse_std":
            if self.svd_type is None:
                self.svd_type = "left" if m >= n else "right"
            # SVD step
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type=self.svd_type)

            if self.svd_type == "left":
                low_rank_grad = torch.matmul(self.ortho_matrix.t()[:, :m], full_rank_grad)
            else:
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t()[:n])
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if low_rank_grad.dim() != 2:
            return

        m, n = low_rank_grad.shape
        if self.svd_type == "right":
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix[:n])
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix[:, :m], low_rank_grad)

        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        # TODO: redo SVD in the next step.
        if matrix.isnan().any():
            print(f"{__file__}: skipping SVD due to NaN matrix")
            return self.ortho_matrix
        try:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
        except _LinAlgError as e:
            print(f"{__file__}: skipping SVD due to {e}")
            return self.ortho_matrix

        # make the smaller matrix always to be orthogonal matrix
        if type == "right":
            B = Vh[:rank, :]

            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == "left":
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == "full":
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError("type should be left, right or full")


class GaLoreAdamW8bit(Optimizer2State):
    r"""Implements Galore, a optimizer-agonistic gradient compression technique on 8-bit AdamW.
    Proposed in `GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection`. It compresses
    gradient via low-rank projection and is claimed to be insensitive to hyperparams like lr.
    https://arxiv.org/abs/2403.03507

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)
        nbits (int): The number of bits of optim states. Only 32 and 8 are supported.
        min_8bit_size (`int`, defaults to 4096):
            The minimum number of elements of the parameter tensors for 8-bit optimization.
        percentile_clipping (`int`, defaults to 100):
            Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
        block_wise (`bool`, defaults to `True`):
            Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
        is_paged (`bool`, defaults to `False`):
            Whether the optimizer is a paged optimizer (handle memory spike via CPU-GPU transfer) or not.
        args (dict, optional): quantization-related arguments. If passed, will override all quantization args above.
    Example:

    """

    def __init__(
        self,
        params,
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        nbits=8,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
        args=None,
    ):
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits=nbits,
            args=args,
            min_8bit_size=min_8bit_size,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise,
            is_paged=is_paged,
        )

        proj_none = all(["rank" not in group for group in self.param_groups])
        if proj_none:
            warnings.warn(
                "Will not apply GaLore as no rank is specified. Or did you forget to? Try get_galore_param_groups"
            )

        # Defaults from the paper
        for group in self.param_groups:
            if "rank" in group:
                group["update_proj_gap"] = group.get("update_proj_gap", 200)
                group["proj_type"] = group.get("proj_type", "std")
                group["scale"] = group.get("scale", 0.25)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if p is self.param_groups[0]["params"][0]:
                    torch.save(p.grad, "grad.pt")
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            group["rank"],
                            scale=group["scale"],
                            update_proj_gap=group["update_proj_gap"],
                            proj_type=group["proj_type"],
                        )

                    if "weight_decay" in group and group["weight_decay"] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group["weight_decay_saved"] = group["weight_decay"]
                        group["weight_decay"] = 0

                    grad = state["projector"].project(p.grad, state["step"])
                    make_low_rank_buffer(p, grad)

                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                # p.grad = p.grad.contiguous() # avoid bitsandbytes update error
                # Prefetch if paged
                self.prefetch_state(p)
                # Adam update step using the buffer
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()

                # GaLore Projection Back
                if "rank" in group:
                    if p is self.param_groups[0]["params"][1]:
                        pass
                    update = state["projector"].project_back(p.data)
                    p.data = p.saved_data.add_(update)

                # apply weight decay
                if "weight_decay_saved" in group:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay_saved"])
                    group["weight_decay"] = group["weight_decay_saved"]
                    del group["weight_decay_saved"]

        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss

    def __del__(self):
        """Avoid buffer memory leak"""
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "saved_data"):
                    del p.saved_data
