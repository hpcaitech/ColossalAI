from galore_torch import GaLoreAdafactor, GaLoreAdamW

from colossalai.logging import get_dist_logger

from .came import CAME
from .cpu_adam import CPUAdam
from .distributed_adafactor import DistributedAdaFactor
from .distributed_came import DistributedCAME
from .distributed_galore import DistGaloreAwamW
from .distributed_lamb import DistributedLamb
from .fused_adam import FusedAdam
from .fused_lamb import FusedLAMB
from .fused_sgd import FusedSGD
from .galore import GaLoreAdamW8bit
from .hybrid_adam import HybridAdam
from .lamb import Lamb
from .lars import Lars

from .adafactor import Adafactor  # noqa

__all__ = [
    "FusedLAMB",
    "FusedAdam",
    "FusedSGD",
    "Lamb",
    "Lars",
    "CPUAdam",
    "HybridAdam",
    "DistributedLamb",
    "DistGaloreAwamW",
    "GaLoreAdamW",
    "GaLoreAdafactor",
    "GaLoreAdamW8bit",
    "CAME",
    "DistributedCAME",
    "Adafactor",
    "DistributedAdaFactor",
]

optim2DistOptim = {
    GaLoreAdamW8bit: DistGaloreAwamW,
    Lamb: DistributedLamb,
    CAME: DistributedCAME,
    Adafactor: DistributedAdaFactor,
}


def cast_to_distributed(optim):
    if optim.__class__ in optim2DistOptim:
        _logger = get_dist_logger()
        _logger.info(f"Converting optimizer {optim.__class__.__name__} to its distributed version.", ranks=[0])

        if isinstance(optim, GaLoreAdamW8bit):
            return optim2DistOptim[GaLoreAdamW8bit](optim.param_groups, args=optim.args)
        return optim2DistOptim[optim.__class__](optim.param_groups)

    return optim
