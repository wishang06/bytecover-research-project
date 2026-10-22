#!/usr/bin/env python3
"""Step 1: check YouTube availability (metadata only, no audio) for a seeded
sample of SHS-100K-2025 train works. Usage:
  python scripts/check_availability.py --csv data/shs-100k/train.csv --n-works 300 --seed 0
Writes data/availability.csv (one row per video) and prints a summary.
Does NOT work around bot checks: a 'Sign in to confirm you're not a bot' error
is recorded as status=blocked and, after --max-blocked in a row, the run stops."""
import argparse, csv, json, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

def check(vid):
    cmd = ["yt-dlp", "--simulate", "--skip-download", "--no-warnings", "--no-playlist",
           "--print", "%(id)s|%(duration)s", f"https://www.youtube.com/watch?v={vid}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return vid, "timeout", ""
    err = (p.stderr or "").strip().splitlines()
    err = err[-1] if err else ""
    if p.returncode == 0:
        return vid, "available", p.stdout.strip()
    low = err.lower()
    if "not a bot" in low or "sign in to confirm" in low or "429" in low or "rate" in low:
        return vid, "blocked", err
    return vid, "unavailable", err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/shs-100k/train.csv")
    ap.add_argument("--n-works", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-blocked", type=int, default=20)
    ap.add_argument("--out", default="data/availability.csv")
    a = ap.parse_args()
    works = defaultdict(list)  # columns: perf_id, work_id, title, artist, yt_id (no header)
    for r in csv.reader(open(a.csv)):
        works[r[1]].append(r)
    ids = sorted(works); random.Random(a.seed).shuffle(ids); sample = ids[:a.n_works]
    rows = [r for w in sample for r in works[w]]
    print(f"{len(sample)} works, {len(rows)} videos", flush=True)
    results, consec_blocked = {}, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (vid, st, info) in enumerate(ex.map(check, [r[4] for r in rows])):
            results[vid] = (st, info)
            consec_blocked = consec_blocked + 1 if st == "blocked" else 0
            if i % 200 == 0: print(i, st, info[:120], flush=True)
            if consec_blocked >= a.max_blocked:
                print("STOP: blocked/rate-limited:", info, flush=True); ex.shutdown(cancel_futures=True); break
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["perf_id", "work_id", "title", "artist", "yt_id", "status", "info"])
        for r in rows:
            st, info = results.get(r[4], ("not_checked", ""))
            w.writerow(r[:5] + [st, info])
    checked = [r for r in rows if r[4] in results]
    avail = [r for r in checked if results[r[4]][0] == "available"]
    per_work = defaultdict(int)
    for r in avail: per_work[r[1]] += 1
    summ = dict(works_sampled=len(sample), videos_total=len(rows), videos_checked=len(checked),
                available=len(avail), blocked=sum(results[r[4]][0] == "blocked" for r in checked),
                availability_rate=round(len(avail) / max(1, len(checked)), 4),
                works_ge2=sum(v >= 2 for v in per_work.values()),
                versions_in_ge2_works=sum(v for v in per_work.values() if v >= 2))
    print(json.dumps(summ, indent=1)); json.dump(summ, open(a.out.replace(".csv", "_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
