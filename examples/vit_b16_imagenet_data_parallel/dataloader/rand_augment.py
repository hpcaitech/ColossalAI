import torch
import numpy as np
import torchvision.transforms.functional as TF

_MAX_LEVEL = 10

_HPARAMS = {
    'cutout_const': 40,
    'translate_const': 40,
}

_FILL = tuple([128, 128, 128])
# RGB


def blend(image0, image1, factor):
    # blend image0 with image1
    # we only use this function in the 'color' function
    if factor == 0.0:
        return image0
    if factor == 1.0:
        return image1
    image0 = image0.type(torch.float32)
    image1 = image1.type(torch.float32)
    scaled = (image1 - image0) * factor
    image = image0 + scaled

    if factor > 0.0 and factor < 1.0:
        return image.type(torch.uint8)

    image = torch.clamp(image, 0, 255).type(torch.uint8)
    return image


def autocontrast(image):
    image = TF.autocontrast(image)
    return image


def equalize(image):
    image = TF.equalize(image)
    return image


def rotate(image, degree, fill=_FILL):
    image = TF.rotate(image, angle=degree, fill=fill)
    return image


def posterize(image, bits):
    image = TF.posterize(image, bits)
    return image


def sharpness(image, factor):
    image = TF.adjust_sharpness(image, sharpness_factor=factor)
    return image


def contrast(image, factor):
    image = TF.adjust_contrast(image, factor)
    return image


def brightness(image, factor):
    image = TF.adjust_brightness(image, factor)
    return image


def invert(image):
    return 255-image


def solarize(image, threshold=128):
    return torch.where(image < threshold, image, 255-image)


def solarize_add(image, addition=0, threshold=128):
    add_image = image.long() + addition
    add_image = torch.clamp(add_image, 0, 255).type(torch.uint8)
    return torch.where(image < threshold, add_image, image)


def color(image, factor):
    new_image = TF.rgb_to_grayscale(image, num_output_channels=3)
    return blend(new_image, image, factor=factor)


def shear_x(image, level, fill=_FILL):
    image = TF.affine(image, 0, [0, 0], 1.0, [level, 0], fill=fill)
    return image


def shear_y(image, level, fill=_FILL):
    image = TF.affine(image, 0, [0, 0], 1.0, [0, level], fill=fill)
    return image


def translate_x(image, level, fill=_FILL):
    image = TF.affine(image, 0, [level, 0], 1.0, [0, 0], fill=fill)
    return image


def translate_y(image, level, fill=_FILL):
    image = TF.affine(image, 0, [0, level], 1.0, [0, 0], fill=fill)
    return image


def cutout(image, pad_size, fill=_FILL):
    b, c, h, w = image.shape
    mask = torch.ones((b, c, h, w), dtype=torch.uint8).cuda()
    y = np.random.randint(pad_size, h-pad_size)
    x = np.random.randint(pad_size, w-pad_size)
    for i in range(c):
        mask[:, i, (y-pad_size): (y+pad_size), (x-pad_size): (x+pad_size)] = fill[i]
    image = torch.where(mask == 1, image, mask)
    return image


def _randomly_negate_tensor(level):
    # With 50% prob turn the tensor negative.
    flip = np.random.randint(0, 2)
    final_level = -level if flip else level
    return final_level


def _rotate_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return level


def _shear_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return level


def _translate_level_to_arg(level, translate_const):
    level = (level/_MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return level


def level(hparams):
    return {
        'AutoContrast': lambda level: None,
        'Equalize': lambda level: None,
        'Invert': lambda level: None,
        'Rotate': _rotate_level_to_arg,
        'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4)),
        'Solarize': lambda level: (int((level/_MAX_LEVEL) * 200)),
        'SolarizeAdd': lambda level: (int((level/_MAX_LEVEL) * 110)),
        'Color': lambda level: ((level/_MAX_LEVEL) * 1.8 + 0.1),
        'Contrast': lambda level: ((level/_MAX_LEVEL) * 1.8 + 0.1),
        'Brightness': lambda level: ((level/_MAX_LEVEL) * 1.8 + 0.1),
        'Sharpness': lambda level: ((level/_MAX_LEVEL) * 1.8 + 0.1),
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'Cutout': lambda level: (int((level/_MAX_LEVEL) * hparams['cutout_const'])),
        'TranslateX': lambda level: _translate_level_to_arg(level, hparams['translate_const']),
        'TranslateY': lambda level: _translate_level_to_arg(level, hparams['translate_const']),
    }


AUGMENTS = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}


def RandAugment(image, num_layers=2, magnitude=_MAX_LEVEL, augments=AUGMENTS):
    """Random Augment for images, followed google randaug and the paper(https://arxiv.org/abs/2106.10270)
    :param image: the input image, in tensor format with shape of C, H, W
    :type image: uint8 Tensor
    :num_layers: how many layers will the randaug do, default=2
    :type num_layers: int
    :param magnitude: the magnitude of random augment, default=10
    :type magnitude: int 
    """
    if np.random.random() < 0.5:
        return image
    Choice_Augment = np.random.choice(a=list(augments.keys()),
                                      size=num_layers,
                                      replace=False)
    magnitude = float(magnitude)
    for i in range(num_layers):
        arg = level(_HPARAMS)[Choice_Augment[i]](magnitude)
        if arg is None:
            image = augments[Choice_Augment[i]](image)
        else:
            image = augments[Choice_Augment[i]](image, arg)
    return image
