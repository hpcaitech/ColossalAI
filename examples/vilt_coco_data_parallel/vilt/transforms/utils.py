from torchvision import transforms
from PIL import Image


class MinMaxResize:
    def __init__(self, shorter=800, longer=1333):
        self.min = shorter
        self.max = longer

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# This is simple maximum entropy normalization performed in Inception paper
inception_normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)

# ViT uses simple non-biased inception normalization
# https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py#L132
inception_unnormalize = transforms.Compose(
    [UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)
