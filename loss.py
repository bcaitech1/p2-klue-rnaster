import torch
import torch.nn as nn
import torch.nn.functional as F


def get_criterion(name="ce", **kwargs):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "focal":
        return FocalLoss(**kwargs)
    if name == "ls":
        return LabelSmoothing(**kwargs)
    raise ValueError("incorrect name", name)


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean', **kwargs):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothing(nn.Module):
    def __init__(self, classes=42, smoothing=0.1, dim=-1, **kwargs):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
