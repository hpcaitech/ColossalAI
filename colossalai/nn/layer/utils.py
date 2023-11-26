def divide(numerator, denominator):
    """Only allow exact division.

    Args:
        numerator (int): Numerator of the division.
        denominator (int): Denominator of the division.

    Returns:
        int: the result of exact division.
    """
    assert denominator != 0, "denominator can not be zero"
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator
