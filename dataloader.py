import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, validation_split=0.0, validation_idx=None):
        self.validation_split = validation_split
        self.validation_idx = validation_idx
        self.shuffle = shuffle
        self.n_samples = len(dataset)
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, self.validation_idx)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super(MyDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, validation_split, validation_idx):
        if (validation_split == 0.0) and (validation_idx is None):
            return None, None

        idx_full = np.arange(self.n_samples)
        if validation_idx is None:
            np.random.seed(0)
            np.random.shuffle(idx_full)

            len_valid = int(self.n_samples * validation_split)

            valid_idx = idx_full[0:len_valid]
            train_idx = np.delete(idx_full, np.arange(0, len_valid))
        else:
            # if validation_idx is provided, validation_split is ignored
            valid_idx = validation_idx
            train_idx = np.delete(idx_full, valid_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            print("Number of training data: %d" % len(self.sampler.indices))
            print("Number of validation data: %d" % len(self.valid_sampler.indices))
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
