import torch
import torch.nn as nn

__all__ = ["Accelerator"]

_supported_devices = [
    "cpu",
    "cuda",
    # To be supported
    # 'xpu',
    # 'npu',
    # 'tpu',
]


class Accelerator:
    """
    Accelerator is an abstraction for the hardware device that is used to run the model.

    Args:
        device (str): The device to be used. Currently only support 'cpu' and 'gpu'.
    """

    def __init__(self, device: str):
        self.device = device

        assert (
            self.device in _supported_devices
        ), f"Device {self.device} is not supported yet, supported devices include {_supported_devices}"

    def bind(self):
        """
        Set the default device for the current process.
        """
        if self.device == "cpu":
            pass
        elif self.device == "cuda":
            # TODO(FrankLeeeee): use global environment to check if it is a dist job
            # if is_distributed:
            #     local_rank = EnvTable().get_local_rank()
            #     torch.cuda.set_device(torch.device(f'cuda:{local_rank}'))
            torch.cuda.set_device(torch.device("cuda"))
        else:
            raise ValueError(f"Device {self.device} is not supported yet")

    def configure_model(self, model: nn.Module) -> nn.Module:
        """
        Move the model to the device.

        Args:
            model (nn.Module): The model to be moved.
        """
        model = model.to(torch.device(self.device))
        return model
