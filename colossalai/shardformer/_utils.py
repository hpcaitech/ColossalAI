import re


def get_obj_list_element(obj, a):
    r"""
    Get the element of the list in the object
    """
    re_pattern = r'\[\d+\]'
    prog = re.compile(re_pattern)
    result = prog.search(a)
    if result:
        matched_brackets = result.group()
        matched_index = matched_brackets.replace('[', '')
        matched_index = matched_index.replace(']', '')
        a_ = a.replace(matched_brackets, '')
        container_obj = getattr(obj, a_)
        obj = container_obj[int(matched_index)]
    else:
        obj = getattr(obj, a)
    return obj


def hasattr_(obj, attr: str):
    r"""
    Check whether the object has the multi sublevel attr

    Args:
        obj (object): The object to check
        attr (str): The multi level attr to check
    """
    attrs = attr.split('.')
    for a in attrs:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            return False
    return True


def setattr_(obj, attr: str, value, ignore: bool = False):
    r"""
    Set the object's multi sublevel attr to value, if ignore, ignore when it doesn't exist

    Args:
        obj (object): The object to set
        attr (str): The multi level attr to set
        value (Any): The value to set
        ignore (bool): Whether to ignore when the attr doesn't exist
    """

    attrs = attr.split('.')
    for a in attrs[:-1]:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return
            raise AttributeError(f"Object {obj.__class__.__name__} has no attribute {attr}")
    setattr(obj, attrs[-1], value)


def getattr_(obj, attr: str, ignore: bool = False):
    r"""
    Get the object's multi sublevel attr

    Args:
        obj (object): The object to set
        attr (str): The multi level attr to set
        ignore (bool): Whether to ignore when the attr doesn't exist
    """

    attrs = attr.split('.')
    for a in attrs:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return None
            raise AttributeError(f"Object {obj.__class__.__name__} has no attribute {attr}")
    return obj
