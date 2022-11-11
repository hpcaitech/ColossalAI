from dataclasses import dataclass
from enum import Enum

__all__ = ['SolverOptions']


class SolverPerference(Enum):
    """
    This enum class is to define the solver preference.
    """
    STANDARD = 0
    DP = 1
    TP = 2


class PlaceholderOption(Enum):
    """
    This enum class is to define the placeholder option.
    """
    REPLICATED = 0
    DISTRIBUTED = 1


@dataclass
class SolverOptions:
    """
    SolverOptions is a dataclass used to configure the preferences for the parallel execution plan search.
    """
    solver_perference: SolverPerference = SolverPerference.STANDARD
    placerholder_option: PlaceholderOption = PlaceholderOption.REPLICATED
