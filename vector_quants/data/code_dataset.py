import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DATASET_CONFIGS


class CodeDataset(Dataset):
    def __init__(self, cfg):
        feature_dir = cfg.feature_dir
        label_dir = cfg.label_dir

        self.feature_dir = feature_dir
        self.label_dir = label_dir
        num = DATASET_CONFIGS[cfg.data_set]["num_class"]
        self.feature_files = [f"{i}.npy" for i in range(num)]
        self.label_files = [f"{i}.npy" for i in range(num)]

    def __len__(self):
        assert len(self.feature_files) == len(
            self.label_files
        ), "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature_dir = self.feature_dir
        label_dir = self.label_dir
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        labels = np.load(os.path.join(label_dir, label_file))

        return torch.from_numpy(features), torch.from_numpy(labels)
