import re


def get_obj_list_element(obj, attr: str):
    r"""
    Get the element of the list in the object

    If the attr is a normal attribute, return the attribute of the object.
    If the attr is a index type, return the element of the index in the list, like `layers[0]`.

    Args:
        obj (Object): The object to get
        attr (str): The suffix of the attribute to get

    """
    re_pattern = r"\[\d+\]"
    prog = re.compile(re_pattern)
    result = prog.search(attr)
    if result:
        matched_brackets = result.group()
        matched_index = matched_brackets.replace("[", "")
        matched_index = matched_index.replace("]", "")
        attr_ = attr.replace(matched_brackets, "")
        container_obj = getattr(obj, attr_)
        obj = container_obj[int(matched_index)]
    else:
        obj = getattr(obj, attr)
    return obj


def set_obj_list_element(obj, attr: str, value):
    r"""
    Set the element to value of a list object

    It used like set_obj_list_element(obj, 'layers[0]', new_layer), it will set obj.layers[0] to value

    Args:
        obj (object): The object to set
        attr (str): the string including a list index like `layers[0]`
    """
    re_pattern = r"\[\d+\]"
    prog = re.compile(re_pattern)
    result = prog.search(attr)
    if result:
        matched_brackets = result.group()
        matched_index = matched_brackets.replace("[", "")
        matched_index = matched_index.replace("]", "")
        attr_ = attr.replace(matched_brackets, "")
        container_obj = getattr(obj, attr_)
        container_obj[int(matched_index)] = value
    else:
        setattr(obj, attr, value)


def hasattr_(obj, attr: str):
    r"""
    Check whether the object has the multi sublevel attr

    Args:
        obj (object): The object to check
        attr (str): The multi level attr to check
    """
    attrs = attr.split(".")
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

    attrs = attr.split(".")
    for a in attrs[:-1]:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return
            raise AttributeError(f"Object {obj.__class__.__name__} has no attribute {attr}")
    set_obj_list_element(obj, attrs[-1], value)


def getattr_(obj, attr: str, ignore: bool = False):
    r"""
    Get the object's multi sublevel attr

    Args:
        obj (object): The object to set
        attr (str): The multi level attr to set
        ignore (bool): Whether to ignore when the attr doesn't exist
    """

    attrs = attr.split(".")
    for a in attrs:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return None
            raise AttributeError(f"Object {obj.__class__.__name__} has no attribute {attr}")
    return obj
