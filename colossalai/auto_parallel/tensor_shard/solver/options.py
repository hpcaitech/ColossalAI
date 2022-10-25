from dataclasses import dataclass

__all__ = ['SolverOptions']


@dataclass
class SolverOptions:
    """
    SolverOptions is a dataclass used to configure the preferences for the parallel execution plan search.
    """
    fast: bool = False
