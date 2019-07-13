import numpy as np
from torch.utils.data import Dataset


def load_data(data_path):
    # TODO: check if reshape results in wrong dimension
    # reshape is to give the correct shape for model.py
    x = np.load(data_path)
    n_data, n_frame, width, height, n_channel = x.shape
    x = x.reshape(n_data, n_frame, n_channel, width, height)
    return x


class SpriteDataset(Dataset):
    def __init__(self, data_path, debug):
        print("Load the dataset ...")
        data = load_data(data_path)
        if debug:
            data = data[:10000]

        self.data_path = data_path
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
