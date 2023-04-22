import torch
from torch.utils.data import DistributedSampler


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_samples=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        if num_samples is not None:
            self.num_samples = min(num_samples, len(dataset))
        else:
            self.num_samples = len(dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # Ensure the same random seed for all processes
            if self.epoch is not None:
                seed = self.epoch
            else:
                seed = 0
            torch.manual_seed(seed)

            # Shuffle the indices
            indices = torch.randperm(
                len(self.dataset), generator=torch.Generator().manual_seed(seed)).tolist()

        # Get the start and end indices for the current process
        start_index = self.rank * self.num_samples // self.num_replicas
        end_index = (self.rank + 1) * self.num_samples // self.num_replicas

        return iter(indices[start_index:end_index])
