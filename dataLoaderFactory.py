from abc import ABC, abstractmethod
from typing import Dict, Any

import torchio as tio
from torch.utils.data import DataLoader

from data_processing.subject_folder import SubjectFolder
from utils import Config


class DataLoaderFactory(ABC):
    """Representation to create dataloader object"""

    @abstractmethod
    def getDataLoader(self, dataset: SubjectFolder, batch_size: int, num_workers: int) -> DataLoader:
        """Creates and returns a dataloader"""


class StandardDataLoader(DataLoaderFactory, Config):
    """Create standard dataloader"""

    def __init__(self, sampler, collate_fn):
        self.sampler = sampler
        self.collate_fn = collate_fn

    def getDataLoader(self, dataset: SubjectFolder, batch_size: int, num_workers: int):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=self.sampler(dataset),
            collate_fn=self.collate_fn,
            num_workers=num_workers,
        )

        return dataloader

    def getConfig(self):
        return {}


class PatchDataLoader(DataLoaderFactory, Config):
    """Create patch based dataloader"""

    def __init__(self, max_length: int, samples_per_volume, sampler, collate_fn):
        self.max_length = max_length
        self.samples_per_volume = samples_per_volume
        self.sampler = sampler
        self.collate_fn = collate_fn

    def getDataLoader(self, dataset: SubjectFolder, batch_size: int, num_workers: int):
        queue = tio.Queue(
            dataset,
            max_length=self.max_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=num_workers,
        )
        dataloader = DataLoader(dataset=queue, batch_size=batch_size, collate_fn=self.collate_fn)

        return dataloader

    def getConfig(self):
        return {"max_length": self.max_length, "samples_per_volume": self.samples_per_volume}
