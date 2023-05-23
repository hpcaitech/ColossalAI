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
            obj = getattr(obj, a)
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
            obj = getattr(obj, a)
        except AttributeError:
            if ignore:
                return
            raise AttributeError(f"Object {obj} has no attribute {attr}")
    setattr(obj, attrs[-1], value)


def getattr_(obj, attr: str, ignore: bool = None):
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
            obj = getattr(obj, a)
        except AttributeError:
            if ignore:
                return None
            raise AttributeError(f"Object {obj} has no attribute {attr}")
    return obj
