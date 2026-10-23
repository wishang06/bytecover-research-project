#!/usr/bin/env python3
"""Build data/train_manifest.csv: one row per cached SHS recording.

Joins audio/downloaded.csv with the cqt/shs cache. data/exclude_works.txt is
the catalog leakage audit (13 SHS works that match Covers80). Those works are
outside the seed-0 sample, so every downloaded row is kept and marked
excluded=0. The script fails if a flagged work does appear in the sample.

  uv run python scripts\\make_manifest.py
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def load_excluded(path):
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [i for i in ids if i]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate work ids in {path}")
    return set(ids)


def load_downloads(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["perf_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate perf_id in {path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--downloads", type=Path, default=Path("audio") / "downloaded.csv")
    ap.add_argument("--exclude", type=Path, default=Path("data") / "exclude_works.txt")
    ap.add_argument("--cqt", type=Path, default=Path("cqt") / "shs")
    ap.add_argument("--out", type=Path, default=Path("data") / "train_manifest.csv")
    ap.add_argument("--kept-works", type=int, default=300)
    a = ap.parse_args()

    excluded = load_excluded(a.exclude)
    rows = load_downloads(a.downloads)
    cached = {p.stem for p in a.cqt.glob("*.npy")}
    logged = {r["perf_id"] for r in rows}

    missing_npy = sorted(logged - cached)
    missing_row = sorted(cached - logged)
    print(f"downloaded rows: {len(rows)}")
    print(f"cached npy: {len(cached)}")
    print(f"in csv but no npy: {len(missing_npy)}")
    print(f"npy with no csv row: {len(missing_row)}")
    if missing_npy[:5] or missing_row[:5]:
        print("examples missing npy:", missing_npy[:5])
        print("examples missing row:", missing_row[:5])

    by_work = Counter(r["work_id"] for r in rows)
    overlap = excluded & set(by_work)
    kept = {w: n for w, n in by_work.items() if w not in excluded}
    thin = sorted(w for w, n in kept.items() if n < 2)
    print(f"works in csv: {len(by_work)}")
    print(f"catalog overlap ids listed: {len(excluded)}")
    print(f"catalog overlap ids inside this sample: {sorted(overlap, key=int)}")
    print(f"kept works: {len(kept)}")
    print(f"kept recordings: {sum(kept.values())}")
    print(f"kept works with <2 versions: {len(thin)}")

    ok = (
        not missing_npy
        and not missing_row
        and not overlap
        and len(kept) == a.kept_works
        and not thin
    )
    if not ok:
        print("FAIL")
        sys.exit(1)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["perf_id", "work_id", "title", "artist", "excluded"]
    ordered = sorted(rows, key=lambda r: (int(r["work_id"]), int(r["perf_id"])))
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in ordered:
            w.writerow({
                "perf_id": r["perf_id"],
                "work_id": r["work_id"],
                "title": r["title"],
                "artist": r["artist"],
                "excluded": "1" if r["work_id"] in excluded else "0",
            })
    print(f"wrote {len(ordered)} rows -> {a.out}")
    print("PASS")


if __name__ == "__main__":
    main()
