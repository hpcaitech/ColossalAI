# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
import copy
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.mixed_precision_mixin import (
    BF16MixedPrecisionMixin,
    FP16MixedPrecisionMixin,
    MixedPrecisionMixin,
)
from colossalai.interface import OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.zero.low_level.low_level_strategy import LowLevelOptStrategy, LowLevelOptStrategyBase

from ._utils import calculate_global_norm_from_list, has_inf_or_nan


class LowLevelZeroFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):
    def __init__(
        self,
        group_strategies: List[LowLevelOptStrategyBase],
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
    ) -> None:
        super().__init__(
            initial_scale, min_scale, growth_factor, backoff_factor, growth_interval, hysteresis, max_scale
        )
        self.group_strategies = group_strategies

    def check_local_overflow(self) -> bool:
        for strategy in self.group_strategies:
            for avg_grad in strategy.working_grads:
                if avg_grad is not None and has_inf_or_nan(avg_grad):
                    return True
        return False


class LowLevelZeroOptimizer(OptimizerWrapper):
    def __init__(
        self,
        optimizer: Optimizer,
        group_strategies: List[LowLevelOptStrategyBase] = None,
        initial_scale: int = 2**16,  # grad scaler config
        min_scale: int = 1,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        hysteresis: int = 2,
        max_scale: int = 2**24,
        clip_grad_norm: float = 0.0,  # grad clipping
        verbose: bool = False,
        forced_dtype: Optional[torch.dtype] = None,
        **strategy_kwargs,
    ):
        super(LowLevelZeroOptimizer, self).__init__(optim=optimizer)
        self._dtype = self.optim.param_groups[0]["params"][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group["params"]
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        if len(self.optim.param_groups) == 1 and group_strategies is None:
            group_strategies = [LowLevelOptStrategy(param_group=self.optim.param_groups[0], **strategy_kwargs)]
        elif len(self.optim.param_groups) > 1 and group_strategies is None:
            raise ValueError("group_strategies must be provided when the optimizer has multiple param groups")

        self.param2strategy: Dict[torch.nn.Parameter, LowLevelOptStrategyBase] = {}
        for grp, strategy in zip(self.optim.param_groups, group_strategies):
            assert grp["params"] is strategy.param_group["params"], "param groups should be in the same order"
            for param in strategy.working_param_group:
                self.param2strategy[param] = strategy
        self._group_strategies = group_strategies

        # initialize mixed precision mixin
        self.mixed_precision_mixin: Optional[MixedPrecisionMixin] = None
        if self._dtype is torch.float16:
            self.mixed_precision_mixin = LowLevelZeroFP16MixedPrecisionMixin(
                self._group_strategies,
                initial_scale=initial_scale,
                min_scale=min_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                hysteresis=hysteresis,
                max_scale=max_scale,
            )
        elif self._dtype is torch.bfloat16:
            self.mixed_precision_mixin = BF16MixedPrecisionMixin()

    def backward(self, loss, retain_graph=False):
        for strategy in self._group_strategies:
            strategy.pre_backward(loss, retain_graph)

        if self.mixed_precision_mixin is not None:
            loss = self.mixed_precision_mixin.pre_backward(loss)

        loss.backward(retain_graph=retain_graph)

        for strategy in self._group_strategies:
            strategy.post_backward()

    def state_dict(self) -> Dict:
        """Return a state_dict same with DDP

        Returns:
            Dict: the pytorch form state_dict
        """
        zero_state = dict()
        device = get_accelerator().get_current_device()
        for strategy in self._group_strategies:
            param_group = strategy.param_group
            for param in param_group:
                state = self.optim.state[param]
                zero_state[param] = copy.deepcopy(state)
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and k != "step":
                        param_state = strategy.allgather_optim_state(param, v)
                        zero_state[param][k] = param_state

        states_dict = self._pack_state(zero_state)

        return states_dict

    def load_state_dict(self, state_dict: Dict):
        """Load state dict, requires the state_dict be the pytorch form

        Args:
            state_dict (dict): A pytorch form state_dict
        """
        zero_state_dict = copy.deepcopy(state_dict)
        self.optim.load_state_dict(zero_state_dict)
        for strategy in self._group_strategies:
            strategy.scatter_optim_state(self.optim.state)

    def update_master_params(self, model: nn.Module) -> None:
        """Update master params from working params

        Args:
            model (nn.Module): The model to update master params
        """
        all_working_params = []
        for stategy in self._group_strategies:
            all_working_params.extend(stategy.working_params)
            stategy.update_master_params()
        assert set(map(lambda x: id(x), all_working_params)) == set(
            map(lambda x: id(x), model.parameters())
        ), "model parameters should be the same"

    def step(self, closure=None):
        assert closure is None, "closure is not supported by step()"
        if not self.require_grad_sync:
            return

        if self.mixed_precision_mixin is not None and self.mixed_precision_mixin.should_skip_step():
            if self._verbose:
                self._logger.info(f"Found overflow. Skip step")
            for strategy in self._group_strategies:
                strategy.zero_working_grad()
                strategy.zero_grad()
            return

        # TODO @botbw can be further refactored
        grad_partition_groups = []
        norm_groups = []
        for strategy in self._group_strategies:
            strategy.pre_step()
            grad_partition_groups.extend(strategy.working_grads)
            norm_groups.append(strategy.get_grad_norm())
            strategy.zero_working_grad()

        # unscale and clip grads
        global_norm = calculate_global_norm_from_list(norm_list=norm_groups)
        self._unscale_and_clip_grads(grad_partition_groups, global_norm)

        # update the parameters
        self.optim.step()

        for strategy in self._group_strategies:
            strategy.post_step()

    @property
    def require_grad_sync(self) -> bool:
        flag_set = set()
        for strategy in self._group_strategies:
            flag_set.add(strategy.require_grad_sync)
        assert len(flag_set) == 1, "require_grad_sync should be the same for all strategies"
        return flag_set.pop()

    # this context comes from pytorch DDP
    @contextmanager
    def no_sync(self):
        old_require_grad_sync = self.require_grad_sync
        for strategy in self._group_strategies:
            strategy.require_grad_sync = False
        try:
            yield
        finally:
            for strategy in self._group_strategies:
                strategy.require_grad_sync = old_require_grad_sync

    ##################################################################################

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        # compute combined scale factor for this group
        div_scale = 1.0
        if self.mixed_precision_mixin is not None:
            div_scale = self.mixed_precision_mixin.get_grad_div_scale()

        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            clip = ((total_norm / div_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1:
                div_scale = clip * div_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1.0 / div_scale)

    def _sanity_checks(self):
        assert get_accelerator().name in ["cuda", "npu"], "device is required"
        inv = defaultdict(list)
        for param_group in self.optim.param_groups:
            group_params = param_group["params"]
            for param in group_params:
                inv[param].append(param_group)
                assert (
                    param.dtype == self._dtype
                ), f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"

        for _, grps in inv.items():
            assert (
                len(grps) == 1
            ), "Parameters should only appear in one group, since we assume that each strategy only manages one param group"

    def _pack_state(self, state: Dict) -> Dict:
        # comes from pytorch optimizer.state_dict()
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {id(p): i for i, p in enumerate(group["params"], start_index) if id(p) not in param_mappings}
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.optim.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v for k, v in state.items()}

        return {"state": packed_state, "param_groups": param_groups}

    # another way of doing this is to reassign tensor.grad, however this won't apply for zero-2
    # since the shape doesn't match
    def get_param_grad(self, param):
        strategy = self.param2strategy[param]
        return strategy.get_param_grad(param)
