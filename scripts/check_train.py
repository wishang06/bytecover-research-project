#!/usr/bin/env python3
"""Check the 10-by-3 batches and one Adam step. The scalar is not a loss.

  uv run python scripts\\check_train.py
"""
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bytecover.dataset import CQTClips, load_manifest, load_work_order, subset_work_ids
from bytecover.train import CLIPS_PER_WORK, WORKS_PER_BATCH, PKSampler, check_l4, make_encoder, train


def epoch_batches(dataset, work_ids, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return list(PKSampler(dataset, work_ids, generator))


def works_of(dataset, batch):
    return [dataset.rows[i]["work_id"] for i in batch]


def main():
    order = load_work_order(ROOT / "data" / "work_order.txt")
    rows = load_manifest(ROOT / "data" / "train_manifest.csv")
    for size, n_batches in ((50, 5), (150, 15), (300, 30)):
        work_ids = subset_work_ids(order, size)
        dataset = CQTClips(rows, ROOT / "cqt" / "shs", work_ids)
        first, second = epoch_batches(dataset, work_ids, seed=0), epoch_batches(dataset, work_ids, seed=0)
        other = epoch_batches(dataset, work_ids, seed=1)
        if first != second or first == other or len(first) != n_batches:
            print(f"FAIL size {size} batch count or seed pairing")
            sys.exit(1)
        seen = []
        for batch in first:
            counts = Counter(works_of(dataset, batch))
            if len(batch) != WORKS_PER_BATCH * CLIPS_PER_WORK or set(counts.values()) != {CLIPS_PER_WORK}:
                print(f"FAIL size {size} batch shape {len(batch)} {sorted(counts.values())}")
                sys.exit(1)
            seen.extend(counts)
        if Counter(seen) != {work_id: 1 for work_id in work_ids}:
            print(f"FAIL size {size} does not cover each work once")
            sys.exit(1)
        print(f"size {size}: {len(first)} batches, each work once")

    first = make_encoder(0).conv1.weight.detach()
    second = make_encoder(0).conv1.weight.detach()
    if not torch.equal(first, second):
        print("FAIL paired seeds do not share initial weights")
        sys.exit(1)

    def dummy(f_t, y):
        # A stand-in so the step can run before Andy's loss exists. Mean keeps
        # the gradient small. The sum of f_t is large enough to blow up Adam.
        return f_t.mean()

    log_path = ROOT / "logs" / "train_loss_check.csv"
    log_path.unlink(missing_ok=True)
    try:
        check_l4("Tesla T4")
        print("FAIL T4 was accepted")
        sys.exit(1)
    except RuntimeError:
        pass
    check_l4("NVIDIA L4")
    model, _head = train(
        dummy, rows, ROOT / "cqt" / "shs", order,
        size=50, seed=0, epochs=1, arm="check", log_path=log_path,
    )
    moved = not torch.equal(first, model.conv1.weight.detach().cpu())
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    logged = log_path.read_text(encoding="utf-8").strip().splitlines()
    header, row = logged[0], logged[1]
    print(f"epoch ran weight_moved={moved} weights_finite={finite}")
    print(row)
    if not moved or not finite or len(logged) != 2 or "check,50,0,1," not in row or not row.endswith(",cpu"):
        print("FAIL step")
        sys.exit(1)
    if header != "arm,size,seed,epoch,mean_objective,gpu":
        print("FAIL header", header)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
