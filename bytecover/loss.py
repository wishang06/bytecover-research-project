"""Arm A and Arm B objectives. The encoder is shared and is not defined here.

Both arms use the same BNNeck. It maps f_t to f_c with batch norm, then a
bias-free linear layer classifies f_c. Retrieval uses f_c for both arms.
Cosine distance is applied by the evaluation, not here.

Arm A is that classification loss only. Arm B adds a batch-hard triplet on
f_t, not on f_c, with Euclidean distance and margin 0.3. In a batch of 10
works by 3 clips, each anchor's positive is the farthest other clip of the
same work, and its negative is the nearest clip of any other work.
"""
import torch
from torch import nn
import torch.nn.functional as F

DIM = 2048
MARGIN = 0.3


def make_loss(arm, n_classes):
    if arm == "A":
        return ArmA(n_classes)
    if arm == "B":
        return ArmB(n_classes)
    raise ValueError(f"arm must be 'A' or 'B', got {arm}")


def batch_hard_triplet(f_t, y, margin):
    """Mean hinge over batch-hard triplets. f_t is (N, D), y is (N,)."""
    dist = torch.cdist(f_t, f_t, p=2)
    same = y[:, None].eq(y[None, :])
    eye = torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
    positive = same & ~eye
    if not bool(positive.any(dim=1).all()):
        raise RuntimeError("an anchor has no other clip of its work in the batch")
    hard_pos = dist.masked_fill(~positive, float("-inf")).max(dim=1).values
    hard_neg = dist.masked_fill(same, float("inf")).min(dim=1).values
    return (hard_pos - hard_neg + margin).clamp(min=0).mean()


class _Neck(nn.Module):
    def __init__(self, n_classes, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.classifier = nn.Linear(dim, n_classes, bias=False)

    def embed(self, f_t):
        return self.bn(f_t)

    def classification(self, f_t, y):
        return F.cross_entropy(self.classifier(self.embed(f_t)), y)


class ArmA(_Neck):
    def __init__(self, n_classes, dim=DIM):
        super().__init__(n_classes, dim)

    def forward(self, f_t, y):
        return self.classification(f_t, y)


class ArmB(_Neck):
    def __init__(self, n_classes, dim=DIM, margin=MARGIN):
        super().__init__(n_classes, dim)
        self.margin = margin

    def forward(self, f_t, y):
        return self.classification(f_t, y) + batch_hard_triplet(f_t, y, self.margin)
