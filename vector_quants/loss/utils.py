from vector_quants.data import get_mean_std_from_dataset_name
from vector_quants.utils import logging_info, rescale_image_tensor


def get_post_transform(post_transform_type, data_set="imagenet-1k"):
    if post_transform_type == 0:
        logging_info(f"Post Transform: None")
        post_transform = lambda x: x
    elif post_transform_type == 1:
        logging_info(f"Post Transform: rescale to [0, 1]")
        norm_mean, norm_std = get_mean_std_from_dataset_name(data_set)
        post_transform = lambda x: rescale_image_tensor(x, norm_mean, norm_std)
    elif post_transform_type == -1:
        logging_info(f"Post Transform: rescale from [-1, 1] to [0, 1] for llamagen")
        # 0.5 * x + 0.5
        norm_mean, norm_std = 0.5, 0.5
        post_transform = lambda x: rescale_image_tensor(x, norm_mean, norm_std)
    else:
        logging_info(f"Post Transform: rescale to [0, 1]")
        norm_mean, norm_std = get_mean_std_from_dataset_name(data_set)
        post_transform = lambda x: rescale_image_tensor(x, norm_mean, norm_std)

    return post_transform
