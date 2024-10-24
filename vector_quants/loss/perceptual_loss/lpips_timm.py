import timm
import torch
import torch.nn as nn


class LpipsTimm(nn.Module):
    """
    Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    """

    def __init__(self, model_name):
        super().__init__()

        model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.model = model
        data_config = timm.data.resolve_model_data_config(model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        feature_x = self.model(self.transforms(x))
        feature_y = self.model(self.transforms(y))

        diff_list = [(fx - fy).pow(2).mean() for fx, fy in zip(feature_x, feature_y)]
        diff = torch.stack(diff_list).mean()

        return diff
