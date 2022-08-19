from .operator_handler import OperatorHanlder


class DotHandler(OperatorHanlder):
    """
    A OperatorHandler which deals with the sharding strategies of linear matrix multiplication.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: refactor the dot handler in my local branch to align with the latest main branch
