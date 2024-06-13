import torch
from torchvision import transforms

from vector_quants.data import get_mean_std_from_dataset_name
from vector_quants.utils import logging_info, rescale_image_tensor

transform_rev = transforms.Normalize(
    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
)


def get_revd_perceptual(inputs, recons, perceptual_model):
    return torch.mean(
        perceptual_model(
            transform_rev(inputs.contiguous()), transform_rev(recons.contiguous())
        )
    )


def get_post_transform(post_transform_type, data_set="imagenet-1k"):
    if post_transform_type == 0:
        logging_info(f"Post Transform: None")
        post_transform = lambda x: x
    elif post_transform_type == 1:
        logging_info(f"Post Transform: fsq_pytorch default")
        post_transform = transform_rev
    else:
        logging_info(f"Post Transform: rescale to [0, 1]")
        norm_mean, norm_std = get_mean_std_from_dataset_name(data_set)
        post_transform = lambda x: rescale_image_tensor(x, norm_mean, norm_std)

    return post_transform
