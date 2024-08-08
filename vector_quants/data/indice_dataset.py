# for generation only
from torch.utils.data import Dataset
from .constants import DATASET_CONFIGS

class IndiceDataset(Dataset):
    def __init__(self, cfg_data):
        num_sample = cfg_data.num_sample
        self.num_class = DATASET_CONFIGS[cfg_data.data_set]["num_class"]
        n = (num_sample + self.num_class - 1) // self.num_class
        self.indice_list = (list(range(self.num_class)) * n)[:num_sample]

    def __len__(self):
        return len(self.indice_list)

    def __getitem__(self, idx):
        return self.indice_list[idx]