# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learning rate decay functions."""

import math


class AnnealingLR(object):
    """Anneals the learning rate."""

    def __init__(
        self,
        optimizer,
        max_lr,
        min_lr,
        warmup_steps,
        decay_steps,
        decay_style,
        use_checkpoint_lr_scheduler=True,
        override_lr_scheduler=False,
    ):
        # Class values.
        self.optimizer = optimizer

        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr

        self.warmup_steps = warmup_steps
        self.num_steps = 0
        self.decay_steps = decay_steps
        assert self.decay_steps > 0
        assert self.warmup_steps < self.decay_steps

        self.decay_style = decay_style

        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, "both override and " "use-checkpoint are set."

        # Set the learning rate
        self.step(0)

    def get_lr(self):
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        # Use linear warmup for the initial part.
        if self.warmup_steps > 0 and self.num_steps <= self.warmup_steps:
            return self.max_lr * float(self.num_steps) / float(self.warmup_steps)

        # If the learning rate is constant, just return the initial value.
        if self.decay_style == "constant":
            return self.max_lr

        # For any steps larger than `self.decay_steps`, use `self.min_lr`.
        if self.num_steps > self.decay_steps:
            return self.min_lr

        # If we are done with the warmup period, use the decay style.
        num_steps_ = self.num_steps - self.warmup_steps
        decay_steps_ = self.decay_steps - self.warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = self.max_lr - self.min_lr

        if self.decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception("{} decay style is not supported.".format(self.decay_style))

        return self.min_lr + coeff * delta_lr

    def step(self, increment=1):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def state_dict(self):
        state_dict = {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            "num_steps": self.num_steps,
            "decay_style": self.decay_style,
            "decay_steps": self.decay_steps,
            "min_lr": self.min_lr,
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert cls_value == sd_value, (
                f"AnnealingLR: class input value {cls_value} and checkpoint" f"value {sd_value} for {name} do not match"
            )
        return sd_value

    def load_state_dict(self, sd):
        if "start_lr" in sd:
            max_lr_ = sd["start_lr"]
        else:
            max_lr_ = sd["max_lr"]
        self.max_lr = self._check_and_set(self.max_lr, max_lr_, "learning rate")

        self.min_lr = self._check_and_set(self.min_lr, sd["min_lr"], "minimum learning rate")

        if "warmup_iter" in sd:
            warmup_steps_ = sd["warmup_iter"]
        else:
            warmup_steps_ = sd["warmup_steps"]
        self.warmup_steps = self._check_and_set(self.warmup_steps, warmup_steps_, "warmup iterations")

        if "end_iter" in sd:
            decay_steps_ = sd["end_iter"]
        else:
            decay_steps_ = sd["decay_steps"]
        self.decay_steps = self._check_and_set(self.decay_steps, decay_steps_, "total number of iterations")
        self.decay_style = self._check_and_set(self.decay_style, sd["decay_style"], "decay style")

        if "num_iters" in sd:
            num_steps = sd["num_iters"]
        else:
            num_steps = sd["num_steps"]
        self.step(increment=num_steps)
