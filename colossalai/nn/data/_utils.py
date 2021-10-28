import numpy as np


def pil_img_to_numpy(pil_img):
    """convert a PIL image to numpy nd-array

    :param pil_img: a PIL image
    :type pil_img: PIL.Image
    :return: a nd-array
    :rtype: numpy.ndarray
    """
    np_img = np.array(pil_img)
    np_img = np.rollaxis(np_img, 2)  # HWC to CHW
    return np_img
