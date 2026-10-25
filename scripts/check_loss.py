#!/usr/bin/env python3
"""Check Arm A cross-entropy and Arm B's batch-hard triplet. No audio.

  uv run python scripts\\check_loss.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bytecover.loss import MARGIN, ArmA, ArmB, batch_hard_triplet


def labels(n_works=10, k=3):
    return torch.arange(n_works).repeat_interleave(k)


def main():
    y = labels()
    easy = torch.zeros(30, 2)
    jitter = torch.tensor([[0.0, 0.0], [0.05, 0.0], [-0.05, 0.0]])
    for work in range(10):
        easy[y == work] = jitter + torch.tensor([10.0 * work, 0.0])
    easy_loss = float(batch_hard_triplet(easy, y, MARGIN))
    if easy_loss != 0.0:
        print(f"FAIL easy triplet {easy_loss}")
        sys.exit(1)

    hard = torch.zeros(6, 2)
    hard[0] = torch.tensor([0.0, 0.0])
    hard[1] = torch.tensor([5.0, 0.0])
    hard[2] = torch.tensor([5.0, 0.0])
    hard[3:] = torch.tensor([0.1, 0.0])
    hard_y = torch.tensor([0, 0, 0, 1, 1, 1])
    # Anchor 0: farthest positive is 5, nearest negative is 0.1.
    expected = (5.0 - 0.1 + MARGIN)
    got = float(batch_hard_triplet(hard, hard_y, MARGIN))
    # Anchors at 5 have a zero-distance positive, so their hinge is smaller.
    # The mean must sit strictly between 0 and the anchor-0 hinge.
    if not (0.0 < got < expected):
        print(f"FAIL hard triplet {got}, anchor hinge {expected}")
        sys.exit(1)

    f_t = torch.randn(30, 16, requires_grad=True)
    arm_a = ArmA(n_classes=10, dim=16)
    loss_a = arm_a(f_t, y)
    loss_a.backward()
    arm_b = ArmB(n_classes=10, dim=16)
    f_b = torch.randn(30, 16, requires_grad=True)
    loss_b = arm_b(f_b, y)
    loss_b.backward()
    ok = (
        arm_a.classifier.bias is None
        and arm_b.classifier.bias is None
        and tuple(arm_a.embed(f_t).shape) == (30, 16)
        and torch.isfinite(loss_a)
        and torch.isfinite(loss_b)
        and arm_a.classifier.weight.grad.abs().sum() > 0
        and arm_b.classifier.weight.grad.abs().sum() > 0
        and f_t.grad.abs().sum() > 0
        and f_b.grad.abs().sum() > 0
        and tuple(arm_b.embed(f_b).shape) == (30, 16)
    )
    print(f"easy_triplet {easy_loss} hard_triplet {got:.4f} anchor_hinge {expected:.4f}")
    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
