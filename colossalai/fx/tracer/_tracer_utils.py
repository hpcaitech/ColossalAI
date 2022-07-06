from typing import List, Union, Any
from ..proxy import ColoProxy, MetaDeviceAttribute

__all__ = ['is_element_in_list', 'extract_meta']


def is_element_in_list(elements: Union[List[Any], Any], list_: List[Any]):
    if isinstance(elements, (tuple, list, set)):
        for ele in elements:
            if ele not in list_:
                return False, ele
    else:
        if elements not in list_:
            return False, elements

    return True, None


def extract_meta(*args, **kwargs):

    def _convert(val):
        if isinstance(val, MetaDeviceAttribute):
            return 'meta'
        elif isinstance(val, ColoProxy):
            assert val.meta_data is not None
            return val.meta_data
        return val

    new_args = [_convert(val) for val in args]
    new_kwargs = {k: _convert(v) for k, v in kwargs.items()}
    return new_args, new_kwargs
