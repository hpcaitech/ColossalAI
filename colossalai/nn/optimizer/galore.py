# adapted from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/adamw8bit.py
import torch
from bitsandbytes.optim.optimizer import Optimizer2State


class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type="std"):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        m, n = full_rank_grad.shape  # For ZeRO sharded grads
        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type="right")
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t()[:m])
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type="left")
                low_rank_grad = torch.matmul(self.ortho_matrix.t()[:, :n], full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type="left")
                low_rank_grad = torch.matmul(self.ortho_matrix.t()[:, :n], full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type="right")
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t()[:m])
        return low_rank_grad

    def project_back(self, low_rank_grad):
        m, n = low_rank_grad.shape
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix[:m])
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix[:, :n], low_rank_grad)
        elif self.proj_type == "reverse_std":
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix[:, :n], low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix[:m])
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
        try:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
        except:
            print(matrix)
            print(matrix.mean())
            exit(1)

        # make the smaller matrix always to be orthogonal matrix
        if type == "right":
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]

            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == "left":
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
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


class GaloreAdamW8bit(Optimizer2State):
    r"""Implements Galore, a optimizer-agonistic gradient compression technique on 8-bit Adam.
    Proposed in `GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection`
    https://arxiv.org/abs/2403.03507
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
    ):
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged=is_paged,
        )

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
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            group["rank"], scale=group["scale"], proj_type=group["proj_type"]
                        )

                    if "weight_decay" in group and group["weight_decay"] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group["weight_decay_saved"] = group["weight_decay"]
                        group["weight_decay"] = 0

                    if state["step"] % group["update_proj_gap"] == 0:
                        grad = state["projector"].project(p.grad)

                    # suboptimal implementation
                    p.saved_data = p.data.clone()
                    p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                    p.data.zero_()
                    p.grad = grad

                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()

                # GaLore Projection Back
                if "rank" in group:
                    p.data = p.saved_data.add_(state["projector"].project_back(p.data))

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
