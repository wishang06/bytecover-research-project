"""Training clips from the cached CQTs.

The 300 works stay in the seed-0 order stored in data/work_order.txt. A
training size of 50, 150, or 300 is a prefix of that list, so the smaller
sets are subsets of the larger ones. Training seeds do not change this list.
They belong on the DataLoader shuffle later.

Each saved array is (84, T) with T <= 86. A batch is padded on the right with
zeros to 86 columns. `length` is the real width, and pooling must ignore the
padded columns.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

N_BINS = 84
PAD_FRAMES = 86
SIZES = (50, 150, 300)


def load_manifest(path):
    import csv
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_work_order(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ids = [i for i in ids if i]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate work ids in {path}")
    return ids


def subset_work_ids(order, size):
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size}")
    if len(order) < size:
        raise ValueError(f"work order has {len(order)} ids, need {size}")
    return list(order[:size])


class CQTClips(Dataset):
    """One cached recording per item, restricted to `work_ids` in that order."""

    def __init__(self, rows, cqt_dir, work_ids, pad_to=PAD_FRAMES):
        self.cqt_dir = Path(cqt_dir)
        self.pad_to = pad_to
        self.class_index = {work_id: i for i, work_id in enumerate(work_ids)}
        allowed = set(work_ids)
        self.rows = [
            row for row in rows
            if row["excluded"] == "0" and row["work_id"] in allowed
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        array = np.load(self.cqt_dir / f"{row['perf_id']}.npy")
        if array.ndim != 2 or array.shape[0] != N_BINS:
            raise ValueError(f"{row['perf_id']} has shape {array.shape}, expected ({N_BINS}, T)")
        length = int(array.shape[1])
        if length < 1 or length > self.pad_to:
            raise ValueError(f"{row['perf_id']} width {length} is outside 1..{self.pad_to}")
        image = np.zeros((1, N_BINS, self.pad_to), dtype=np.float32)
        image[0, :, :length] = array.astype(np.float32)
        return {
            "x": torch.from_numpy(image),
            "length": length,
            "y": self.class_index[row["work_id"]],
            "work_id": row["work_id"],
            "perf_id": row["perf_id"],
        }


def collate_clips(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.long),
        "work_id": [item["work_id"] for item in batch],
        "perf_id": [item["perf_id"] for item in batch],
    }
