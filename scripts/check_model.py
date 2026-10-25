#!/usr/bin/env python3
"""Check ResNet50-IBN + GeM on one real batch.

  uv run python scripts\\check_model.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bytecover.dataset import CQTClips, collate_clips, load_manifest, load_work_order, subset_work_ids
from bytecover.model import ResNet50IBN


def main():
    order = load_work_order(ROOT / "data" / "work_order.txt")
    rows = load_manifest(ROOT / "data" / "train_manifest.csv")
    dataset = CQTClips(rows, ROOT / "cqt" / "shs", subset_work_ids(order, 50))
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_clips)))

    model = ResNet50IBN().eval()
    with torch.no_grad():
        feature_map = model.forward_map(batch["x"])
        embedding = model(batch["x"], batch["length"])
        widths = model.valid_widths(torch.tensor([86, 76, 15]), feature_map.shape[-1])

    print(f"map shape: {tuple(feature_map.shape)}")
    print(f"embedding shape: {tuple(embedding.shape)}")
    print(f"valid columns for lengths 86, 76, 15: {widths.tolist()}")
    ok = (
        tuple(feature_map.shape) == (2, 2048, 6, 6)
        and tuple(embedding.shape) == (2, 2048)
        and torch.isfinite(embedding).all()
        and float(embedding.abs().sum()) > 0
        and widths.tolist() == [6, 5, 1]
    )
    valid0 = int(model.valid_widths(batch["length"][:1], feature_map.shape[-1])[0])
    manual = _gem_prefix(feature_map[0], valid0)
    if not torch.allclose(embedding[0], manual, rtol=1e-4, atol=1e-5):
        print("FAIL gem does not match the masked formula")
        sys.exit(1)
    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def _gem_prefix(feature_map, valid_w):
    """Equation (1) on the valid time prefix of one clip. feature_map is (C, H, W)."""
    power = 3.0
    region = feature_map[:, :, :valid_w].clamp(min=1e-6)
    return region.pow(power).mean(dim=(1, 2)).pow(1.0 / power)


if __name__ == "__main__":
    main()
