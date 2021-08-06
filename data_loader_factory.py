from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio

from utils import Config, dont_collate


class DataLoaderFactory(ABC, Config):
    """Representation to create dataloader object"""

    @abstractmethod
    def get_data_loader(self, dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
        """Creates and returns a dataloader"""


class StandardDataLoader(DataLoaderFactory):
    """Create standard dataloader"""

    def __init__(self, sampler):
        self.sampler = sampler

    def get_data_loader(self, dataset: Dataset, batch_size: int, num_workers: int):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=self.sampler(dataset),
            collate_fn=dont_collate,
            num_workers=num_workers,
        )

        return dataloader


class PatchDataLoader(DataLoaderFactory):
    """Create patch based dataloader"""

    def __init__(self, max_length: int, samples_per_volume, sampler):
        self.max_length = max_length
        self.samples_per_volume = samples_per_volume
        self.sampler = sampler

    def get_data_loader(self, dataset: tio.SubjectsDataset, batch_size: int, num_workers: int):
        queue = tio.Queue(
            dataset,
            max_length=self.max_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=num_workers,
        )
        dataloader = DataLoader(dataset=queue, batch_size=batch_size, collate_fn=dont_collate)

        return dataloader
