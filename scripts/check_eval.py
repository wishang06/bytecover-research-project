#!/usr/bin/env python3
"""Check Covers80 pairing and the mAP arithmetic. No trained model.

  uv run python scripts\\check_eval.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bytecover.evaluate import average_precision, cover_pairs, mean_average_precision


def main():
    paths, song_id = cover_pairs(
        ROOT / "data" / "covers80" / "list1.txt",
        ROOT / "data" / "covers80" / "list2.txt",
        ROOT / "cqt" / "covers80",
    )
    cached = list((ROOT / "cqt" / "covers80").glob("*.npy"))
    if len(paths) != 160 or len(set(song_id)) != 80 or len(set(paths)) != 160:
        print(f"FAIL pairs {len(paths)} songs {len(set(song_id))}")
        sys.exit(1)
    if average_precision([False, False, True]) != 1 / 3:
        print("FAIL average precision")
        sys.exit(1)

    def direction(degrees):
        rad = torch.tensor(degrees * torch.pi / 180)
        return torch.stack((torch.cos(rad), torch.sin(rad)))

    # Cosine similarity ignores magnitude, so these tests use direction.
    perfect = torch.stack((direction(0), direction(5), direction(90), direction(95)))
    # Directions 0, 180, 30, 90. Hand ranks give AP 1/3, 1/3, 1/2, 1.
    mixed = torch.stack((direction(0), direction(180), direction(30), direction(90)))
    perfect_map = mean_average_precision(perfect, [0, 0, 1, 1])
    mixed_map = mean_average_precision(mixed, [0, 0, 1, 1])
    expected = (1 / 3 + 1 / 3 + 1 / 2 + 1) / 4
    print(f"official tracks {len(paths)} cached {len(cached)} left out {len(cached) - len(paths)}")
    print(f"perfect mAP {perfect_map:.3f} mixed mAP {mixed_map:.3f}")
    if perfect_map != 1.0 or abs(mixed_map - expected) > 1e-5:
        print(f"FAIL mAP expected {expected}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
