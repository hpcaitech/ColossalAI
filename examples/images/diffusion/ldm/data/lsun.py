import os

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# This class is used to create a dataset of images from LSUN dataset for training
class LSUNBase(Dataset):
    def __init__(
        self,
        txt_file,  # path to the text file containing the list of image paths
        data_root,  # root directory of the LSUN dataset
        size=None,  # the size of images to resize to
        interpolation="bicubic",  # interpolation method to be used while resizing
        flip_p=0.5,  # probability of random horizontal flipping
    ):
        self.data_paths = txt_file  # store path to text file containing list of images
        self.data_root = data_root  # store path to root directory of the dataset
        with open(self.data_paths, "r") as f:  # open and read the text file
            self.image_paths = f.read().splitlines()  # read the lines of the file and store as list
        self._length = len(self.image_paths)  # store the number of images

        # create dictionary to hold image path information
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
        }

        # set the image size to be resized
        self.size = size
        # set the interpolation method for resizing the image
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        # randomly flip the image horizontally with a given probability
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        # return the length of dataset
        return self._length

    def __getitem__(self, i):
        # get the image path for the given index
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        # convert it to RGB format
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing

        img = np.array(image).astype(np.uint8)  # convert image to numpy array
        crop = min(img.shape[0], img.shape[1])  # crop the image to a square shape
        (
            h,
            w,
        ) = (
            img.shape[0],
            img.shape[1],
        )  # get the height and width of image
        img = img[
            (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
        ]  # crop the image to a square shape

        image = Image.fromarray(img)  # create an image from numpy array
        if self.size is not None:  # if image size is provided, resize the image
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)  # flip the image horizontally with the given probability
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)  # normalize the image values and convert to float32
        return example  # return the example dictionary containing the image and its file paths


# A dataset class for LSUN Churches training set.
# It initializes by calling the constructor of LSUNBase class and passing the appropriate arguments.
# The text file containing the paths to the images and the root directory where the images are stored are passed as arguments. Any additional keyword arguments passed to this class will be forwarded to the constructor of the parent class.
class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


# A dataset class for LSUN Churches validation set.
# It is similar to LSUNChurchesTrain except that it uses a different text file and sets the flip probability to zero by default.
class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(
            txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches", flip_p=flip_p, **kwargs
        )


# A dataset class for LSUN Bedrooms training set.
# It initializes by calling the constructor of LSUNBase class and passing the appropriate arguments.
class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


# A dataset class for LSUN Bedrooms validation set.
# It is similar to LSUNBedroomsTrain except that it uses a different text file and sets the flip probability to zero by default.
class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms", flip_p=flip_p, **kwargs)


# A dataset class for LSUN Cats training set.
# It initializes by calling the constructor of LSUNBase class and passing the appropriate arguments.
# The text file containing the paths to the images and the root directory where the images are stored are passed as arguments.
class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


# A dataset class for LSUN Cats validation set.
# It is similar to LSUNCatsTrain except that it uses a different text file and sets the flip probability to zero by default.
class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats", flip_p=flip_p, **kwargs)
