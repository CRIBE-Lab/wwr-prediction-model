"""Losses and metrics for semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: list[float], ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index)
        valid = target != self.ignore_index
        clean = target.clone()
        clean[~valid] = 0
        truth = F.one_hot(clean, self.num_classes).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)
        valid4 = valid.unsqueeze(1)
        intersection = (probs * truth * valid4).sum((0, 2, 3))
        denominator = ((probs + truth) * valid4).sum((0, 2, 3))
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        return 0.5 * ce + 0.5 * (1.0 - dice.mean())


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    valid = target != ignore_index
    encoded = target[valid] * num_classes + prediction[valid]
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics(confusion: torch.Tensor) -> dict[str, float]:
    matrix = confusion.float()
    true_positive = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - true_positive
    iou = true_positive / union.clamp_min(1)
    accuracy = true_positive.sum() / matrix.sum().clamp_min(1)
    return {"mean_iou": float(iou.mean()), "pixel_accuracy": float(accuracy)}

