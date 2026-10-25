#!/usr/bin/env python3
"""Check the nested work order and one padded training batch.

  uv run python scripts\\check_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bytecover.dataset import PAD_FRAMES, SIZES, CQTClips, collate_clips, load_manifest, load_work_order, subset_work_ids


def main():
    order = load_work_order(ROOT / "data" / "work_order.txt")
    rows = load_manifest(ROOT / "data" / "train_manifest.csv")
    kept = {row["work_id"] for row in rows if row["excluded"] == "0"}
    if set(order) != kept or len(order) != SIZES[-1]:
        print(f"FAIL work order {len(order)} vs kept works {len(kept)}")
        sys.exit(1)
    nested = [subset_work_ids(order, size) for size in SIZES]
    if not (nested[0] == nested[1][:50] == nested[2][:50] and nested[1] == nested[2][:150]):
        print("FAIL sizes are not nested prefixes")
        sys.exit(1)

    works = nested[0]
    dataset = CQTClips(rows, ROOT / "cqt" / "shs", works)
    present = {row["work_id"] for row in dataset.rows}
    if present != set(works):
        print(f"FAIL size-50 dataset works {len(present)} != 50")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_clips)
    batch = next(iter(loader))
    if tuple(batch["x"].shape) != (4, 1, 84, PAD_FRAMES):
        print(f"FAIL batch shape {tuple(batch['x'].shape)}")
        sys.exit(1)

    for i, perf_id in enumerate(batch["perf_id"]):
        raw = np.load(ROOT / "cqt" / "shs" / f"{perf_id}.npy")
        length = int(batch["length"][i])
        work_id = batch["work_id"][i]
        if length != raw.shape[1] or work_id not in works:
            print(f"FAIL {perf_id} length {length} raw {raw.shape}")
            sys.exit(1)
        if int(batch["y"][i]) != works.index(work_id):
            print(f"FAIL label for {work_id}")
            sys.exit(1)
        got = batch["x"][i, 0, :, :length]
        if not torch.equal(got, torch.from_numpy(raw.astype(np.float32))):
            print(f"FAIL values {perf_id}")
            sys.exit(1)
        if length < PAD_FRAMES and not torch.equal(batch["x"][i, :, :, length:], torch.zeros(1, 84, PAD_FRAMES - length)):
            print(f"FAIL padding {perf_id}")
            sys.exit(1)

    short = next(row for row in dataset.rows if np.load(ROOT / "cqt" / "shs" / f"{row['perf_id']}.npy").shape[1] < PAD_FRAMES)
    item = dataset[dataset.rows.index(short)]
    print(f"work order: {len(order)}")
    print(f"size 50 recordings: {len(dataset)}")
    print(f"batch shape: {tuple(batch['x'].shape)}")
    print(f"short clip {short['perf_id']} length {item['length']} pad {PAD_FRAMES}")
    print("PASS")


if __name__ == "__main__":
    main()
