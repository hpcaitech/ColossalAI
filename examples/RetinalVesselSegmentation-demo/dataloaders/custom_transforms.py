import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import imshow, imsave
from scipy.ndimage.interpolation import map_coordinates
import cv2
from scipy import ndimage


def to_multilabel(pre_mask, classes = 2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [0, 1]
    mask[pre_mask == 2] = [1, 1]
    return mask


class add_salt_pepper_noise():
    def __call__(self, sample):
        image = sample['image']
        X_imgs_copy = np.asarray(image).copy()

        salt_vs_pepper = 0.2
        amount = 0.004

        num_salt = np.ceil(amount * X_imgs_copy.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy.size * (1.0 - salt_vs_pepper))

        seed = random.random()
        if seed > 0.75:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 1
        elif seed > 0.5:
            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 0
        sample['image'] = X_imgs_copy
        return sample

class adjust_light():
    def __call__(self, sample):
        image = sample['image']
        seed = random.random()
        if seed > 0.5:
            gamma = random.random() * 3 + 0.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)
            sample['image'] = image
        return sample


class eraser():
    def __call__(self, sample, s_l=0.02, s_h=0.06, r_1=0.3, r_2=0.6, v_l=0, v_h=255, pixel_level=False):
        image = sample['image']
        img_h, img_w, img_c = image.shape


        if random.random() > 0.5:
            return sample

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        image[top:top + h, left:left + w, :] = c
        sample['image'] = image
        return sample

class elastic_transform():
    """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """

    # def __init__(self):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        alpha = image.size[1] * 2
        sigma = image.size[1] * 0.08
        random_state = None
        seed = random.random()
        if seed > 0.5:
            # print(image.size)
            assert len(image.size) == 2

            if random_state is None:
                random_state = np.random.RandomState(None)

            shape = image.size[0:2]
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            transformed_image = np.zeros([image.size[0], image.size[1], 3])
            transformed_label = np.zeros([image.size[0], image.size[1]])
            for i in range(3):
                # print(i)
                transformed_image[:, :, i] = map_coordinates(np.array(image)[:, :, i], indices, order=1).reshape(shape)
                # break
            if label is not None:
                transformed_label[:, :] = map_coordinates(np.array(label)[:, :], indices, order=1, mode='nearest').reshape(shape)
            else:
                transformed_label = None
            transformed_image = transformed_image.astype(np.uint8)

            if label is not None:
                transformed_label = transformed_label.astype(np.uint8)
            sample['image'] = Image.fromarray(transformed_image)
            sample['label'] = transformed_label
        return sample






class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        # print(img.size)
        w, h = img.size
        if self.padding > 0 or w < self.size[0] or h < self.size[1]:
            padding = np.maximum(self.padding,np.maximum((self.size[0]-w)//2+5,(self.size[1]-h)//2+5))
            img = ImageOps.expand(img, border=padding, fill=0)
            mask = ImageOps.expand(mask, border=padding, fill=255)

        assert img.width == mask.width
        assert img.height == mask.height
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask,
                    'img_name': sample['img_name'],
                    'dc': sample['dc']}
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        # print(img.size)
        sample['image'] = img
        sample['label'] = mask
        return sample


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.width == mask.width
        # assert img.height == mask.height
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        # y1 = int(round((h - th) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}


class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        sample['image'] = img
        sample['label'] = mask
        return sample



class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']

        assert img.width == mask.width
        assert img.height == mask.height
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'img_name': name}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.width == mask.width
        assert img.height == mask.height
        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask,
                    'img_name': sample['img_name']}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        assert img.width == mask.width
        assert img.height == mask.height
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask,
                        'img_name': name}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, size=512):
        self.degree = random.randint(1, 4) * 90
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        seed = random.random()
        if seed > 0.5:
            rotate_degree = self.degree
            img = img.rotate(rotate_degree, Image.BILINEAR, expand=0)
            mask = mask.rotate(rotate_degree, Image.NEAREST, expand=255)

            sample['image'] = img
            sample['label'] = mask
        return sample


class RandomScaleCrop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # print(img.size)
        assert img.width == mask.width
        assert img.height == mask.height

        seed = random.random()
        if seed > 0.5:
            w = int(random.uniform(1, 1.5) * img.size[0])
            h = int(random.uniform(1, 1.5) * img.size[1])

            img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
            sample['image'] = img
            sample['label'] = mask

        return self.crop(sample)


class ResizeImg(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        assert img.width == mask.width
        assert img.height == mask.height

        img = img.resize((self.size, self.size))
        # mask = mask.resize((self.size, self.size))

        sample = {'image': img, 'label': mask, 'img_name': name}
        return sample


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        assert img.width == mask.width
        assert img.height == mask.height

        img = img.resize((self.size, self.size))
        mask = mask.resize((self.size, self.size))

        sample = {'image': img, 'label': mask, 'img_name': name}
        return sample


# class RandomScale(object):
#     def __init__(self, limit):
#         self.limit = limit
#
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         assert img.width == mask.width
#         assert img.height == mask.height
#
#         scale = random.uniform(self.limit[0], self.limit[1])
#         w = int(scale * img.size[0])
#         h = int(scale * img.size[1])
#
#         img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
#
#         return {'image': img, 'label': mask, 'img_name': sample['img_name']}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}


class GetBoundary(object):
    def __init__(self, width = 5):
        self.width = width
    def __call__(self, mask):
        cup = mask[:, :, 0]
        disc = mask[:, :, 1]
        dila_cup = ndimage.binary_dilation(cup, iterations=self.width).astype(cup.dtype)
        eros_cup = ndimage.binary_erosion(cup, iterations=self.width).astype(cup.dtype)
        dila_disc= ndimage.binary_dilation(disc, iterations=self.width).astype(disc.dtype)
        eros_disc= ndimage.binary_erosion(disc, iterations=self.width).astype(disc.dtype)
        cup = dila_cup + eros_cup
        disc = dila_disc + eros_disc
        cup[cup==2]=0
        disc[disc==2]=0
        size = mask.shape
        # boundary = np.zers(size[0:2])
        boundary = (cup + disc) > 0
        return boundary.astype(np.uint8)


class Normalize_tf(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
        self.get_boundary = GetBoundary()

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        __mask = np.array(sample['label']).astype(np.uint8)
        img /= 127.5
        img -= 1.0
        _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
        _mask[__mask > 200] = 255
        # index = np.where(__mask > 50 and __mask < 201)
        _mask[(__mask > 50) & (__mask < 201)] = 128
        _mask[(__mask > 50) & (__mask < 201)] = 128

        __mask[_mask == 0] = 2
        __mask[_mask == 255] = 0
        __mask[_mask == 128] = 1

        mask = to_multilabel(__mask)
        sample['image'] = img
        sample['label'] = mask
        return sample


class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}

def ToMultiLabel(dc):
    new_dc = np.zeros([3])
    for i in range(new_dc.shape[0]):
        if i == dc:
            new_dc[i] = 1
            return new_dc

def SoftLable(label):
    new_label = label.copy()
    label = list(label)
    index = label.index(1)
    new_label[index] = 0.8+random.random()*0.2
    accelarate = new_label[index]
    for i in range(len(label)):
        if i != index:
            if i == len(label) - 1:
                new_label[i] = 1 - accelarate
            else:
                new_label[i] = random.random()*(1-accelarate)
                accelarate += new_label[i]
    return new_label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        map = np.array(sample['label']).astype(np.uint8).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        map = torch.from_numpy(map).float()
        sample['image']=img
        sample['label']=map
        domain_code = torch.from_numpy(SoftLable(ToMultiLabel(sample['dc']))).float()
        sample['dc'] = domain_code
        return sample