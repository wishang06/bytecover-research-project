#!/usr/bin/env python3
"""Check grid bookkeeping. Does not train.

  uv run python scripts\\check_grid.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bytecover.grid import drop_loss_rows, is_done, run_grid, write_rows
from bytecover.train import LOSS_COLUMNS


def main():
    try:
        run_grid("m", "w", "c", "k", "a", "b", ROOT / "logs" / "grid_should_not_run", device="cpu")
        print("FAIL cpu grid was accepted")
        sys.exit(1)
    except RuntimeError as error:
        if "L4" not in str(error):
            print(f"FAIL unexpected error {error}")
            sys.exit(1)

    folder = ROOT / "logs" / "grid_check"
    folder.mkdir(parents=True, exist_ok=True)
    loss_path = folder / "train_loss.csv"
    runs_path = folder / "runs.csv"
    write_rows(loss_path, LOSS_COLUMNS, [
        {"arm": "A", "size": "50", "seed": "0", "epoch": "1", "mean_objective": "1.0", "gpu": "cpu"},
        {"arm": "B", "size": "50", "seed": "0", "epoch": "1", "mean_objective": "2.0", "gpu": "cpu"},
    ])
    write_rows(runs_path, ["arm", "size", "seed", "map", "gpu", "epochs"], [
        {"arm": "A", "size": "50", "seed": "0", "map": "0.100000", "gpu": "NVIDIA L4", "epochs": "100"},
    ])
    drop_loss_rows(loss_path, "A", 50, 0)
    kept = loss_path.read_text(encoding="utf-8")
    if "A,50,0" in kept or "B,50,0" not in kept:
        print(f"FAIL scrub {kept}")
        sys.exit(1)
    if not is_done(runs_path, "A", 50, 0) or is_done(runs_path, "B", 50, 0):
        print("FAIL done flags")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
