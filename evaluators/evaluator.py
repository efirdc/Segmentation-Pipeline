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
            'mean': lambda x: torch.mean(Evaluator.fix_tensor(x), dim=dim),
            'median': lambda x: torch.median(Evaluator.fix_tensor(x), dim=dim).values,
            'mode': lambda x: torch.mode(Evaluator.fix_tensor(x), dim=dim).values,
            'std': lambda x: torch.std(Evaluator.fix_tensor(x), dim=dim),
            'min': lambda x: torch.min(Evaluator.fix_tensor(x), dim=dim).values,
            'max': lambda x: torch.max(Evaluator.fix_tensor(x), dim=dim).values,
        }

    @staticmethod
    def fix_tensor(x):
        x = x[x.isfinite()]
        if x.shape[0] == 0:
            return torch.tensor([0.])
        return x