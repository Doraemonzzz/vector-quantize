# credit to https://github.com/facebookresearch/deit/blob/main/augment.py

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
import random

from PIL import ImageFilter, ImageOps
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2, activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def new_data_aug_generator(args=None):
    img_size = args.image_size
    remove_random_resized_crop = args.src
    primary_tfl = []
    scale = (0.08, 1.0)
    interpolation = "bicubic"
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
        ]

    secondary_tfl = [
        transforms.RandomChoice(
            [gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]
        )
    ]

    if args.color_jitter is not None and not args.color_jitter == 0:
        secondary_tfl.append(
            transforms.ColorJitter(
                args.color_jitter, args.color_jitter, args.color_jitter
            )
        )

    return primary_tfl + secondary_tfl
