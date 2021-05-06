import torch
import torch.nn.functional as F


def one_hot(x, num_classes):
    return F.one_hot(x.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def saturate_probabilities(x):
    return one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1])