def hasattr_(obj, attr: str):
    attrs = attr.split('.')
    for a in attrs:
        try:
            obj = getattr(obj, a)
        except AttributeError:
            return False
    return True

def setattr_(obj, attr: str, value, ingore: bool=False):
    attrs = attr.split('.')
    for a in attrs[:-1]:
        try:
            obj = getattr(obj, a)
        except AttributeError:
            if ingore:
                 return
            raise AttributeError(f"Object {obj} has no attribute {a}")
    setattr(obj, attrs[-1], value)
