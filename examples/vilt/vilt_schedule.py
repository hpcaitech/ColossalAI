import colossalai
import torch

class viltSchedule(colossalai.engine.schedule.NonPipelineSchedule):
    @staticmethod
    def _call_engine_criterion(engine, outputs, labels):
        # assert isinstance(outputs, (torch.Tensor, list, tuple)
        #                   ), f'Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}'
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )
        if isinstance(labels, torch.Tensor):
            return engine.criterion(*outputs, labels)
        else:
            return engine.criterion(outputs)