import torch
from torch import nn
import torch.nn.functional as F


def one_hot(x, num_classes):
    return F.one_hot(x.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def saturate_probabilities(x):
    return one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1])


class HybridLogisticDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, logistic_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.logistic_weights = logistic_weights

    def forward(self, prediction, target):
        N, C, W, H, D = prediction.shape
        spatial_dims = (2, 3, 4)

        eps = 1e-8

        overlap = torch.sum(prediction * target, dim=spatial_dims)
        total = torch.sum(target * target, dim=spatial_dims) + torch.sum(prediction * prediction, dim=spatial_dims)
        dice_coeffs = 2 * overlap / (total + eps)

        logistic = torch.mean(target * (torch.log(prediction + eps) - eps), dim=spatial_dims)
        if self.logistic_weights is not None:
            logistic_weights = torch.tensor(self.logistic_weights)[None]
            logistic_weights = logistic_weights.to(logistic.device)
            logistic = logistic * logistic_weights

        logistic_loss = -logistic
        dice_loss = 1 - dice_coeffs

        logistic_loss = logistic_loss * (1. - self.dice_weight)
        dice_loss = dice_loss * self.dice_weight

        hybrid = logistic_loss + dice_loss

        return {
            'loss': torch.mean(hybrid),
            "dice_loss": torch.mean(dice_loss),
            "logistic_loss": torch.mean(logistic_loss)
        }


def make_sequential(label):
    label = label.clone()
    current_ids = [i.item() for i in label.unique(sorted=True)]
    for i in range(len(current_ids)):
        if current_ids[i] != i:
            label[label == current_ids[i]] = i
    return label


def dice_metric(label_a, label_b, num_classes):
    spatial_dims = (2, 3, 4)

    label_a = make_sequential(label_a)
    label_b = make_sequential(label_b)

    label_a = one_hot(label_a, num_classes + 1)
    label_b = one_hot(label_b, num_classes + 1)

    overlap = (label_a * label_b).sum(dim=spatial_dims)
    total = label_a.sum(dim=spatial_dims) + label_b.sum(dim=spatial_dims)
    dice_coeffs = 2 * overlap / total

    return dice_coeffs


def dice_validation(prediction, target, label_values, prefix=None):
    dice_scores = dice_metric(prediction, target, len(label_values)).mean(dim=0)
    names = list(label_values.keys())
    names.sort(key=lambda name: label_values[name])
    names = [name.replace("_", " ").title() for name in names]
    if prefix is not None:
        names = [prefix + " " + name for name in names]
    out = {name: dice_scores[i + 1] for i, name in enumerate(names)}
    return out
