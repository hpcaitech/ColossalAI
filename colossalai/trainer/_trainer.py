#!/usr/bin/env python
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright [yyyy] [name of copyright owner]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# -*- encoding: utf-8 -*-

from typing import Union, List
from colossalai.context.parallel_mode import ParallelMode

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from colossalai.core import global_context as gpc

from colossalai.engine import Engine
from colossalai.engine.schedule import NonPipelineSchedule, BaseSchedule
from colossalai.logging import DistributedLogger
from colossalai.utils import MultiTimer
from colossalai.utils import is_dp_rank_0, is_tp_rank_0, is_no_pp_or_last_stage
from colossalai.trainer.hooks import BaseHook
from colossalai.trainer.ophooks import BaseOpHook, register_ophooks_recursively


class Trainer:
    """This a class tending for easy deployments of users' training and evaluation instead of
    writing their own scripts. It is similar with ``ignite.engine`` and ``keras.engine``, but is
    called `Trainer`.

    :param engine: Engine responsible for the process function
    :type engine: :class:`Engine`
    :param schedule: Schedule responsible for forward and backward steps
    :type schedule: :class:`BaseSchedule`, optional
    :param timer: Timer used to monitor the whole training
    :type timer: :class:`MultiTimer`, optional
    :param logger: Logger used to record the whole training
    :type logger: :class:`colossalai.logging.DistributedLogger`, optional
    """
    def __init__(
            self,
            engine: Engine,
            schedule: BaseSchedule = None,
            timer: MultiTimer = None,
            logger: DistributedLogger = None,
    ):
        # training-ralated params
        self._engine = engine
        self._max_epochs = 0
        self._cur_epoch = 0
        self._max_steps = 0
        self._cur_step = 0
        self._steps_per_epoch = 0

        # misc params
        self._logger = logger
        self._verbose = logger is not None

        # hooks can store states in this dict, and could be consumed by other hooks
        self.states = dict()

        # build hooks
        self.hooks = list()

        # multi-timer for time benchmarking
        self._timer = timer

        # set schedule which specifies the training iteration for the engine
        if schedule is None:
            schedule = NonPipelineSchedule()
        if (gpc.is_initialized(ParallelMode.PIPELINE)
                and gpc.get_world_size(ParallelMode.PIPELINE) > 1):
            assert not isinstance(
                schedule, NonPipelineSchedule
            ), "NonPipelineSchedule cannot be used for pipeline parallel training, please use PipelineSchedule instead."
        self._schedule = schedule
        self._schedule.pre_processing(engine)

    def register_ophooks(self, ophook_list):
        """Register the ophooks for the model"""
        register_ophooks_recursively(self._engine._model, ophook_list)

    @property
    def cur_epoch(self):
        """Returns the index of the current epoch."""
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, epoch: int):
        """Set how many epochs have been processed."""
        # allow setter for training resumption
        self._cur_epoch = epoch

    @property
    def cur_step(self):
        """Returns how many iteration steps have been processed."""
        return self._cur_step

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch

    @property
    def engine(self):
        return self._engine

    @property
    def schedule(self):
        return self._schedule

    def _set_current_step(self, epoch: int):
        """Sets current step number.

        :param epoch: Step number to be set
        :type epoch: int
        """
        self._cur_step = epoch * self._steps_per_epoch

    def _call_timer(self, action: str, item: str, *args, **kwargs) -> None:
        """Call timer funciton with a given timer name.

        :param action: Function to be called on timer
        :type action: str
        :param item: Name of the timer
        :type item: str
        :param args: args used for action function
        :param kwargs: kwargs used for action function
        """

        if self._timer is not None:
            getattr(self._timer, action)(item, *args, **kwargs)

    def _reset_states(self) -> None:
        """Clear trainer states"""
        self.states = dict()

    def _call_hooks(self, func, output=None):
        """Calls specific hooks in the current time point.

        :param func: A string represents the time point
        :param output: Output of the model after running a iteration or None in any other time points
        :type func: str
        :type output: optional
        """
        # Only after iter hook will receive output
        for hook in self.hooks:
            if output is None:
                getattr(hook, func)(self)
            else:
                getattr(hook, func)(self, *output)

    @staticmethod
    def _should_display_progress(display_progress: bool):
        """Only display progress on DP rank 0, TP rank 0 and PP last rank"""
        return (display_progress and is_dp_rank_0() and is_tp_rank_0()
                and is_no_pp_or_last_stage())

    def _train_epoch(
            self,
            train_dataloader: DataLoader,
            epoch: int = None,
            display_progress: bool = False,
            return_output_label: bool = True,
    ):
        # set training state
        self._engine.train()
        data_iter = iter(train_dataloader)
        progress = range(self._steps_per_epoch)
        if display_progress:
            if epoch is None:
                progress = tqdm(progress, desc="[Train]")
            else:
                progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

        self._call_hooks("before_train_epoch")
        self._call_timer(action="start", item="Train-epoch")
        for i in progress:
            self._call_hooks("before_train_iter")
            self._call_timer(action="start", item="Train-step")

            # run 1 training step
            self.engine.zero_grad()
            logits, label, loss = self.schedule.forward_backward_step(
                self.engine,
                data_iter,
                forward_only=False,
                return_loss=True,
                return_output_label=return_output_label,
            )
            self.engine.step()
            self._call_timer(action="stop",
                             item="Train-step",
                             keep_in_history=True)
            self._call_hooks("after_train_iter", output=(logits, label, loss))

            self._cur_step += 1

            if display_progress:
                if "step_metrics" in self.states:
                    progress.set_postfix(**self.states["step_metrics"])

            # stop when max iter is reached
            if self._exceed_max_step():
                break

        self._call_timer(action="stop",
                         item="Train-epoch",
                         keep_in_history=True)
        self._call_hooks("after_train_epoch")
        self._call_timer(action="reset", item="Train-epoch")

    def _eval(
            self,
            test_dataloader: DataLoader,
            epoch: int = None,
            display_progress: bool = False,
            return_output_label: bool = True,
    ):
        # switch engine status
        self._engine.eval()

        data_iter = iter(test_dataloader)
        num_steps = len(test_dataloader)

        self._call_hooks("before_test")
        # prepare progress bar
        progress = range(num_steps)
        if display_progress:
            desc = "Evaluation"
            if epoch is not None:
                desc = "[Epoch %d / Test]" % epoch
            progress = tqdm(progress, desc=desc)

        self._call_hooks("before_test_epoch")
        self._call_timer(action="start", item="Test-epoch")
        with torch.no_grad():
            for _ in progress:
                self._call_hooks("before_test_iter")
                self._call_timer(action="start", item="Test-step")
                logits, label, loss = self.schedule.forward_backward_step(
                    self.engine,
                    data_iter,
                    forward_only=True,
                    return_loss=True,
                    return_output_label=return_output_label,
                )
                self._call_timer(action="stop",
                                 item="Test-step",
                                 keep_in_history=True)
                self._call_hooks("after_test_iter",
                                 output=(logits, label, loss))

                if display_progress:
                    if "step_metrics" in self.states:
                        progress.set_postfix(**self.states["step_metrics"])

        self._call_timer(action="stop",
                         item="Test-epoch",
                         keep_in_history=True)
        self._call_hooks("after_test_epoch")
        self._call_hooks("after_test")
        self._call_timer(action="reset", item="Test-step")
        self._call_timer(action="reset", item="Test-epoch")

    def _exceed_max_step(self):
        return self._max_steps is not None and self._cur_step >= self._max_steps

    def fit(
            self,
            train_dataloader: DataLoader,
            epochs: int,
            max_steps: int = None,
            test_dataloader: DataLoader = None,
            test_interval: int = 1,
            hooks: List[BaseHook] = None,
            ophook_list: List[BaseOpHook] = [],
            display_progress: bool = False,
            return_output_label: bool = True,
    ):
        """Trains the model to fit training data.

        :param train_dataloader: DataLoader in training
        :param epochs: Maximum number of epoches
        :param max_steps: Maximum number of running iterations
        :param test_dataloader: DataLoader in testing
        :param test_interval: Interval of testing
        :param hooks: A list of hooks used in training
        :param display_progress: If True, the training progress will be printed
        :param return_output_label: If True, the output of model and the label will be returned

        :type train_dataloader: DataLoader
        :type epochs: int
        :type max_steps: int, optional
        :type test_dataloader: DataLoader, optional
        :type test_interval: int, optional
        :type hooks: list, optional
        :type display_progress: bool, optional
        :type return_output_label: bool, optional
        """

        # set epochs and steps, consider gradient accumulation
        self._steps_per_epoch = len(train_dataloader)
        self._max_steps = max_steps
        self._max_epochs = epochs

        # check if testing is required
        should_test = False
        if test_dataloader is not None:
            should_test = True

        display_progress = self._should_display_progress(display_progress)

        # reset hooks
        self._reset_states()
        if hooks is not None:
            assert isinstance(
                hooks, list
            ), f"expected argument hooks be to list, but got {type(hooks)}"
        else:
            hooks = []
        self.hooks = hooks
        self.hooks.sort(key=lambda hook: hook.priority)
        if self._verbose:
            for hook in self.hooks:
                self._logger.info(
                    f"Using {hook.__class__.__name__} for training, priority = {hook.priority}",
                    ranks=[0],
                )
            self._logger.info(
                "Lower value means higher priority for calling hook function",
                ranks=[0])
        self._call_hooks("after_hook_is_attached")

        # start train
        self.register_ophooks(ophook_list)
        self._engine.train()
        self._call_hooks("before_train")

        # recover step value if resuming training
        last_epoch = self._cur_epoch
        if self.cur_epoch != 0:
            self._set_current_step(last_epoch)

        for epoch in range(last_epoch, epochs):
            # train for one epoch
            self._train_epoch(
                train_dataloader=train_dataloader,
                epoch=epoch,
                display_progress=display_progress,
                return_output_label=return_output_label,
            )

            # start eval
            if should_test and epoch % test_interval == 0:
                self._eval(
                    test_dataloader=test_dataloader,
                    display_progress=display_progress,
                    epoch=epoch,
                    return_output_label=return_output_label,
                )

            self._cur_epoch += 1

            # check for termination
            if self._exceed_max_step():
                self._logger.info(
                    f"Max number of steps {max_steps} has been reached, training is stopped automatically",
                    ranks=[0],
                )
                break
        self._call_hooks("after_train")
        self._call_timer("reset", "Train-epoch")

    def evaluate(
            self,
            test_dataloader: DataLoader,
            hooks: List[BaseHook] = None,
            display_progress: bool = False,
            return_output_label: bool = True,
    ):
        """Evaluates the model with testing data.

        :param test_dataloader: DataLoader in testing
        :param hooks: A list of hooks used in evaluation
        :param display_progress: If True, the evaluation progress will be printed
        :param return_output_label: If True, the output of model and the label will be returned

        :type test_dataloader: DataLoader
        :type hooks: list, optional
        :type display_progress: bool, optional
        :type return_output_label: bool
        """
        # set display
        display_progress = self._should_display_progress(display_progress)

        # reset hooks
        self._reset_states()
        if hooks is not None:
            assert isinstance(
                hooks, list
            ), f"expected argument hooks be to list, but got {type(hooks)}"
        else:
            hooks = []
        self.hooks = hooks
        self.hooks.sort(key=lambda hook: hook.priority)
        if self._verbose:
            for hook in self.hooks:
                self._logger.info(
                    f"Using {hook.__class__.__name__} for training, priority = {hook.priority}",
                    ranks=[0],
                )
            self._logger.info(
                "Lower value means higher priority for calling hook function",
                ranks=[0])
        self._call_hooks("after_hook_is_attached")

        # eval
        self._eval(
            test_dataloader=test_dataloader,
            display_progress=display_progress,
            return_output_label=return_output_label,
        )

    def predict(self, data: Union[Tensor, List[Tensor]]):
        """Uses trained model to make a prediction for a tensor or a tensor list.

        :param data: Data as the input
        :type data: Union[Tensor, List[Tensor]
        :return: The output of model as the prediction
        :rtype: Tensor
        """
        # predict without labels
        if isinstance(data, (list, tuple)):
            assert isinstance(data[0], Tensor)
        else:
            assert isinstance(data, Tensor)
        self._engine.eval()

        # prepare a list of (data, label) to make it iterable
        # for compatibility with schedule
        simple_dataloader = [(data, None)]
        data_iter = iter(simple_dataloader)
        output, _, _ = self.schedule.forward_backward_step(self.engine,
                                                           data_iter,
                                                           forward_only=True,
                                                           return_loss=False)
        return output
