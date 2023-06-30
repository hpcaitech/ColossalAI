from typing import Any, Dict, List, Optional, Tuple

from colossalai.lazy import LazyTensor
from torch import Tensor
from torch.nn import Module, Parameter

from colossalai.pipeline.stage_manager import PipelineStageManager


class Policy:
    def __init__(self, stage_manager: PipelineStageManager) -> None:
        self.stage_manager = stage_manager

    def setup_model(self, module: Module) -> Tuple[Dict[str, Parameter], Dict[str, Tensor]]:
        """Setup model for pipeline parallel

        Args:
            module (Module): Module to be setup

        Returns:
            Tuple[Dict[str, Parameter], Dict[str, Tensor]]: Hold parameters and buffers
        """
        hold_params = set()
        hold_buffers = set()

        def init_layer(layer: Module):
            for p in layer.parameters():
                if isinstance(p, LazyTensor):
                    p.materialize()
                p.data = p.cuda()
                hold_params.add(p)
            for b in layer.buffers():
                if isinstance(b, LazyTensor):
                    b.materialize()
                b.data = b.cuda()
                hold_buffers.add(b)

        hold_layers = self.get_hold_layers(module)

        for layer in hold_layers:
            init_layer(layer)

        hold_params_dict = {}
        hold_buffers_dict = {}

        # release other tensors
        for n, p in module.named_parameters():
            if p in hold_params:
                hold_params_dict[n] = p
            else:
                if isinstance(p, LazyTensor):
                    p.materialize()
                p.data = p.cuda()
                p.storage().resize_(0)
        for n, b in module.named_buffers():
            if b in hold_buffers:
                hold_buffers_dict[n] = b
            else:
                if isinstance(b, LazyTensor):
                    b.materialize()
                b.data = b.cuda()
                # FIXME(ver217): use meta tensor may be better
                b.storage().resize_(0)
        return hold_params_dict, hold_buffers_dict

    def replace_forward(self, module: Module) -> None:
        """Replace module forward in place. This method should be implemented by subclass. The output of internal layers must be a dict

        Args:
            module (Module): _description_
        """
        raise NotImplementedError

    def get_hold_layers(self, module: Module) -> List[Module]:
        """Get layers that should be hold in current stage. This method should be implemented by subclass.

        Args:
            module (Module): Module to be setup

        Returns:
            List[Module]: List of layers that should be hold in current stage
        """
        raise NotImplementedError

    def get_shared_params(self, module: Module) -> List[Dict[int, Tensor]]:
        """Get parameters that should be shared across stages. This method should be implemented by subclass.

        Args:
            module (Module): Module to be setup

        Returns:
            List[Module]: List of parameters that should be shared across stages. E.g. [{0: module.model.embed_tokens.weight, 3: module.lm_head.weight}]
        """
        raise NotImplementedError

    def parallelize_model(self, module: Module) -> Tuple[Dict[str, Parameter], Dict[str, Tensor], List[Dict[int, Tensor]]]:
        """Parallelize model for pipeline parallel

        Args:
            module (Module): Module to be setup

        Returns:
            Tuple[Dict[str, Parameter], Dict[str, Tensor], List[Dict[int, Tensor]]]: Hold parameters, buffers and shared parameters
        """
        hold_params, hold_buffers = self.setup_model(module)
        self.replace_forward(module)
        shared_params = self.get_shared_params(module)
        return hold_params, hold_buffers, shared_params
