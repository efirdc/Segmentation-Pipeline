import torch
from torch import nn
import torch.nn.functional as F


def one_hot(x, num_classes):
    return F.one_hot(x.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def saturate_probabilities(x):
    return one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1])


class HybridLogisticDiceLoss(nn.Module):
    def __init__(self, ignore_background=False, dice_weight=0.5):
        super().__init__()
        self.ignore_background = ignore_background
        self.dice_weight = dice_weight

    def forward(self, prediction, target):
        N, C, W, H, D = prediction.shape
        spatial_dims = (2, 3, 4)
        target = one_hot(target, C)

        overlap = torch.sum(prediction * target, dim=spatial_dims)
        total = torch.sum(target * target, dim=spatial_dims) + torch.sum(prediction * prediction, dim=spatial_dims)
        dice_coeffs = 2 * overlap / total

        eps = 1e-8
        logistic = torch.mean(target * (torch.log(prediction + eps) - eps), dim=spatial_dims)

        logistic_loss = -logistic
        dice_loss = 1 - dice_coeffs

        hybrid = ((1. - self.dice_weight) * logistic_loss + self.dice_weight * dice_loss) * 2.

        return torch.mean(hybrid)


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
