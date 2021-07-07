from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torchio as tio


class Evaluator(ABC):
    @abstractmethod
    def __call__(self, subjects: Sequence[tio.Subject]) -> dict:
        raise NotImplementedError()

    @staticmethod
    def get_summary_stat_funcs(dim: int = 0):
        return {
            'mean': lambda x: torch.mean(x, dim=dim),
            'median': lambda x: torch.median(x, dim=dim).values,
            'mode': lambda x: torch.mode(x, dim=dim).values,
            'std': lambda x: torch.std(x, dim=dim),
            'min': lambda x: torch.min(x, dim=dim).values,
            'max': lambda x: torch.max(x, dim=dim).values,
        }
