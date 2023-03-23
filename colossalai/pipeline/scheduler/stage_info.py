from typing import Any, Dict, Tuple


class StageInput:
    __slots__ = ('minibatch_id', 'args', 'kwargs')
    minibatch_id: int
    args: Tuple[Any]
    kwargs: Dict[str, Any]

    def __init__(self, minibatch_id, args, kwargs) -> None:
        for attr_name in self.__slots__:
            setattr(self, attr_name, locals()[attr_name])


class StageOutput:
    __slots__ = ('minibatch_id', 'output')
    minibatch_id: int
    output: Any

    def __init__(self, minibatch_id, output) -> None:
        for attr_name in self.__slots__:
            setattr(self, attr_name, locals()[attr_name])
