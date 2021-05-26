from torch.utils.data import Dataset
from typing import Dict, Sequence
from utils import slice_volume


class ValidationImage:
    def __init__(
            self,
            training_interval: int,
            name: str,
            channel_id: int,
            slice_id: int,
            legend: bool,
            ncol: int,
            subject_names: Sequence[str],
    ):
        self.training_interval = training_interval
        self.name = name
        self.channel_id = channel_id
        self.slice_id = slice_id
        self.legend = legend
        self.ncol = ncol
        self.subject_names = subject_names


class SegmentationValidator:
    def __init__(
            self,
            val_datasets: Sequence[Dataset],
            val_images: Sequence[ValidationImage]
    ):
        self.val_datasets = val_datasets
        self.val_image = val_images