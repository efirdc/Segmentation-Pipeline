from abc import ABC, abstractmethod
from typing import Sequence, Union

import torchio as tio

from ..utils import auto_str


class Evaluator(ABC):
    @abstractmethod
    def __call__(self, subjects: Union[Sequence[tio.Subject], tio.SubjectsDataset]) -> dict:
        raise NotImplementedError()

    def __repr__(self):
        return auto_str(self)
