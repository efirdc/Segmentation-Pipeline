from typing import Sequence

import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = models

    def forward(self, x):

        output = []

        for model in self.models:
            output.append(model(x))

        output = torch.stack(output)

        averaged = torch.mean(output, dim=0)

        return averaged
