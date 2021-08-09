import torch
from torch import nn
import torch.nn.functional as F


class HybridLogisticDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, logistic_class_weights=None, square_dice=True):
        super().__init__()
        self.dice_weight = dice_weight
        self.logistic_class_weights = logistic_class_weights
        self.square_dice = square_dice

    def forward(self, prediction, target):
        spatial_dims = (2, 3, 4)
        eps = 1e-8

        overlap = torch.sum(prediction * target, dim=spatial_dims)
        if self.square_dice:
            total = torch.sum(target * target, dim=spatial_dims) + torch.sum(prediction * prediction, dim=spatial_dims)
        else:
            total = torch.sum(target, dim=spatial_dims) + torch.sum(prediction, dim=spatial_dims)
        dice_coeffs = 2 * overlap / (total + eps)

        # Shift prediction from [0, 1] range to [eps, 1] range
        prediction_safe = (prediction + eps) / (1 + eps)

        logistic = torch.mean(target * torch.log(prediction_safe), dim=spatial_dims)
        if self.logistic_class_weights is not None:
            weights = torch.tensor(self.logistic_class_weights)[None]
            weights = weights.to(logistic.device)
            logistic = logistic * weights

        logistic_loss = torch.mean(-logistic)
        dice_loss = torch.mean(1 - dice_coeffs)

        t = self.dice_weight
        hybrid_loss = (1. - t) * logistic_loss + t * dice_loss

        return {
            'loss': hybrid_loss,
            "dice_loss": dice_loss,
            "logistic_loss": logistic_loss,
        }
