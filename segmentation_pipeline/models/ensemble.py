from typing import Sequence
import itertools

import torch
from torch import nn
import torch.nn.functional as F


def parse_strategy(strategy: str):
    strategies = ('mean', 'majority')
    if strategy not in strategies:
        raise ValueError(f"Ensembling strategy must be one of {strategies} not {strategy}")
    return strategy


def apply_strategy(
        predictions: Sequence[torch.Tensor],
        strategy: str
):
    # Stack the predictions into a (E, N, C, ...) tensor
    predictions = torch.stack(predictions)

    if strategy == 'mean':
        return torch.mean(predictions, dim=0)

    elif strategy == 'majority':
        C = predictions.shape[2]
        y = torch.argmax(predictions, dim=2)       # (E, N, ...)
        y = torch.mode(y, dim=0).values                   # (N, ...)
        y = F.one_hot(y, num_classes=C)            # (N, ..., C)
        y = y.moveaxis(-1, 1)                      # (N, C, ...)
        return y

    else:
        raise RuntimeError(f"Invalid prediction strategy {strategy}")


class EnsembleModels(nn.Module):
    def __init__(self, models: Sequence[nn.Module], strategy: str = 'mean'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = parse_strategy(strategy)

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        prediction = apply_strategy(predictions, self.strategy)
        return prediction


class EnsembleFlips(nn.Module):
    def __init__(self, model: nn.Module, strategy: str = 'mean', spatial_dims: Sequence[int] = (2, 3, 4)):
        super().__init__()
        self.model = model
        self.strategy = parse_strategy(strategy)
        self.spatial_dims = spatial_dims

        self.flips = []
        for order in range(len(self.spatial_dims) + 1):
            self.flips += list(itertools.combinations(self.spatial_dims, order))

    def forward(self, x):
        predictions = []

        for flip in self.flips:
            x_flipped = x.flip(flip)
            y = self.model(x_flipped)
            y_inverse = y.flip(flip)
            predictions.append(y_inverse)

        prediction = apply_strategy(predictions, self.strategy)
        return prediction


class EnsembleOrientations(nn.Module):
    def __init__(self, model: nn.Module, strategy: str = 'mean'):
        super().__init__()
        self.model = model
        self.strategy = parse_strategy(strategy)

        # Generate all permutations and flips for the spatial_dims
        # TODO: Use case where spatial_dims are a parameter? Permutations are tricky if spatial_dims=(2, 4)
        spatial_dims = (2, 3, 4)
        self.permutations = list(itertools.permutations(spatial_dims))
        self.flips = []
        for order in range(len(spatial_dims) + 1):
            self.flips += list(itertools.combinations(spatial_dims, order))

    def forward(self, x):
        predictions = []
        for permutation in self.permutations:
            inverse_permutation = torch.argsort(torch.tensor(permutation)) + 2
            inverse_permutation = tuple(inverse_permutation.tolist())

            x_permuted = x.permute(0, 1, *permutation)
            for flip in self.flips:
                x_flipped = x_permuted.flip(flip)
                y = self.model(x_flipped)

                y_inverse = y.flip(flip).permute(0, 1, *inverse_permutation)
                predictions.append(y_inverse)

        prediction = apply_strategy(predictions, self.strategy)
        return prediction
