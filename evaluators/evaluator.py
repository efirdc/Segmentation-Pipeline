from abc import ABC, abstractmethod
from typing import Sequence

import torchio as tio


class Evaluator(ABC):
    @abstractmethod
    def __call__(self, subjects: Sequence[tio.Subject]) -> dict:
        raise NotImplementedError()
