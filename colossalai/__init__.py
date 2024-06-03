from . import accelerator
from .initialize import launch, launch_from_openmpi, launch_from_slurm, launch_from_torch

try:
    # .version will be created by setup.py
    from .version import __version__
except ModuleNotFoundError:
    # this will only happen if the user did not run `pip install`
    # and directly set PYTHONPATH to use Colossal-AI which is a bad practice
    __version__ = "0.0.0"
    print("please install Colossal-AI from https://www.colossalai.org/download or from source")

__all__ = ["launch", "launch_from_openmpi", "launch_from_slurm", "launch_from_torch", "__version__"]
