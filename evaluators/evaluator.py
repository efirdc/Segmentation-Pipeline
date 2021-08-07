from abc import ABC, abstractmethod
from typing import Sequence, Union

import torchio as tio

from utils import Config


class Evaluator(ABC, Config):
    @abstractmethod
    def __call__(self, subjects: Union[Sequence[tio.Subject], tio.SubjectsDataset]) -> dict:
        raise NotImplementedError()
