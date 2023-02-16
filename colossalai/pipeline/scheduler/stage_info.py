from typing import Any, Dict, Tuple


class StageInput:
    __slots__ = ('args', 'kwargs')
    args: Tuple[Any]
    kwargs: Dict[str, Any]

    def __init__(self, args, kwargs) -> None:
        for attr_name in self.__slots__:
            setattr(self, attr_name, locals()[attr_name])


class StageOutput:
    __slots__ = ('output')
    output: Any

    def __init__(self, output) -> None:
        for attr_name in self.__slots__:
            setattr(self, attr_name, locals()[attr_name])
