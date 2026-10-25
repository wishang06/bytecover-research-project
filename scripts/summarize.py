#!/usr/bin/env python3
"""Paired B minus A comparison from results/runs.csv.

At each training size the five seeds are pairs. Delta is mAP_B - mAP_A.
The interval is a 95 percent paired t interval with 4 degrees of freedom.
Covers80 is not read here. This script only aggregates finished runs.

  uv run python scripts\\summarize.py
"""
import csv
import sys
from pathlib import Path

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs.csv"
OUT = ROOT / "results" / "summary.csv"
SIZES = (50, 150, 300)
COLUMNS = [
    "size", "n", "mean_map_a", "mean_map_b", "mean_delta",
    "sd_delta", "df", "t_crit", "ci_low", "ci_high",
]


def load_pairs(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_size = {size: {} for size in SIZES}
    for row in rows:
        size = int(row["size"])
        seed = int(row["seed"])
        if size not in by_size:
            raise SystemExit(f"unexpected size {size}")
        by_size[size].setdefault(seed, {})[row["arm"]] = float(row["map"])
    pairs = {}
    for size, seeds in by_size.items():
        if sorted(seeds) != [0, 1, 2, 3, 4]:
            raise SystemExit(f"size {size} seeds {sorted(seeds)}")
        for seed, arms in seeds.items():
            if set(arms) != {"A", "B"}:
                raise SystemExit(f"size {size} seed {seed} arms {sorted(arms)}")
        pairs[size] = [(seeds[s]["A"], seeds[s]["B"]) for s in range(5)]
    return pairs


def interval(deltas):
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    sd = var ** 0.5
    df = n - 1
    crit = float(student_t.ppf(0.975, df))
    half = crit * sd / (n ** 0.5)
    return mean, sd, df, crit, mean - half, mean + half


def main():
    pairs = load_pairs(RUNS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for size in SIZES:
            a_scores = [a for a, _ in pairs[size]]
            b_scores = [b for _, b in pairs[size]]
            deltas = [b - a for a, b in pairs[size]]
            mean, sd, df, crit, low, high = interval(deltas)
            writer.writerow({
                "size": size,
                "n": len(deltas),
                "mean_map_a": f"{sum(a_scores) / len(a_scores):.6f}",
                "mean_map_b": f"{sum(b_scores) / len(b_scores):.6f}",
                "mean_delta": f"{mean:.6f}",
                "sd_delta": f"{sd:.6f}",
                "df": df,
                "t_crit": f"{crit:.6f}",
                "ci_low": f"{low:.6f}",
                "ci_high": f"{high:.6f}",
            })
            print(
                f"size {size}: A {sum(a_scores)/5:.4f}  B {sum(b_scores)/5:.4f}  "
                f"delta {mean:.4f}  95% [{low:.4f}, {high:.4f}]  "
                f"deltas {[round(d, 4) for d in deltas]}"
            )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
