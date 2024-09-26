import timm
import torch
import torch.nn as nn


class LpipsTimm(nn.Module):
    """
    Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, model_name):
        super().__init__()

        model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.model = model
        data_config = timm.data.resolve_model_data_config(model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, x, y):
        feature_x = self.model(self.transforms(x))
        feature_y = self.model(self.transforms(y))

        diff_list = [(fx - fy).pow(2).mean() for fx, fy in zip(feature_x, feature_y)]
        diff = torch.stack(diff_list).mean()

        return diff
