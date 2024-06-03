class LayoutException(Exception):
    pass


class DuplicatedShardingDimensionError(LayoutException):
    pass


class ShardingNotDivisibleError(LayoutException):
    pass


class ShardingOutOfIndexError(LayoutException):
    pass
