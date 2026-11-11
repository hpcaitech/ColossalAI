from galore_torch import GaLoreAdafactor, GaLoreAdamW

from colossalai.logging import get_dist_logger

from .came import CAME
from .cpu_adam import CPUAdam
from .distributed_adafactor import DistributedAdaFactor
from .distributed_came import DistributedCAME
from .distributed_galore import DistGaloreAdamW, DistGaloreAwamW
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
    "DistGaloreAdamW",
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
    GaLoreAdamW8bit: DistGaloreAdamW,
    Lamb: DistributedLamb,
    CAME: DistributedCAME,
    Adafactor: DistributedAdaFactor,
}


def cast_to_distributed(optim, lr_scheduler=None):
    """Return the distributed version of ``optim`` if it has one, else ``optim`` itself.

    The distributed version is a new optimizer on the ``param_groups`` of ``optim``. ``lr_scheduler``, if built on
    ``optim``, is pointed at the new optimizer: left on ``optim``, it only reaches the new one through the param_group
    dicts they share, which ends when a loaded checkpoint replaces the new optimizer's ``param_groups``.
    """
    if optim.__class__ in optim2DistOptim:
        _logger = get_dist_logger()
        _logger.info(f"Converting optimizer {optim.__class__.__name__} to its distributed version.", ranks=[0])

        if isinstance(optim, GaLoreAdamW8bit):
            dist_optim = optim2DistOptim[GaLoreAdamW8bit](optim.param_groups, args=optim.args)
        else:
            dist_optim = optim2DistOptim[optim.__class__](optim.param_groups)
        if lr_scheduler is not None:
            _move_lr_scheduler(lr_scheduler, optim, dist_optim)
        return dist_optim

    return optim


def _move_lr_scheduler(lr_scheduler, old_optim, new_optim):
    if getattr(lr_scheduler, "optimizer", None) is old_optim:
        lr_scheduler.optimizer = new_optim
    # schedulers that step other schedulers: torch's SequentialLR and ChainedScheduler keep them in `_schedulers`,
    # ColossalAI's two-stage ones (warmup, delay) in `after_scheduler`
    nested = [*getattr(lr_scheduler, "_schedulers", ()), getattr(lr_scheduler, "after_scheduler", None)]
    for scheduler in nested:
        if scheduler is not None:
            _move_lr_scheduler(scheduler, old_optim, new_optim)
