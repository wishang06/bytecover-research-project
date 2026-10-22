#!/usr/bin/env python3
"""Download SHS-100K-2025 audio for the 300-work seed-0 sample. Runs on Windows/macOS/Linux
with plain Python + yt-dlp + ffmpeg on PATH (no bash).

  python scripts\\download_audio.py --csv data\\shs-100k\\train.csv --out audio

- samples the same 300 works as check_availability.py (seed 0)
- per work, tries versions in random order, stops at --per-work (5) successes
- writes <out>/<perf_id>.mp3: mono, 22050 Hz, first 200 s
- sleeps --sleep seconds (+ jitter) between requests; resumable (skips existing files
  and versions already logged as failed unless --retry-failed)
- failures -> <out>/failures.csv; successes -> <out>/downloaded.csv
- stops on repeated bot checks / rate limits; it does NOT work around them.
--cookies-from-browser BROWSER is OFF by default. Using it sends your browser's YouTube
login to yt-dlp; it is Andy's choice whether to use it (and with which account).
"""
import argparse, csv, os, random, shutil, subprocess, sys, time
from collections import defaultdict

BLOCK_MARKERS = ("not a bot", "sign in to confirm", "http error 429", "too many requests")

def load_sample(path, n_works, seed):
    works = defaultdict(list)  # perf_id, work_id, title, artist, yt_id (no header)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            works[r[1]].append(r)
    ids = sorted(works); random.Random(seed).shuffle(ids)
    return [(w, works[w]) for w in ids[:n_works]]

def read_set(path, col):
    if not os.path.exists(path): return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r[col] for r in csv.DictReader(f)}

def append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)

def download(yt_id, perf_id, out, a):
    tmpl = os.path.join(out, f"{perf_id}.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "-q",
           "-f", "bestaudio/best", "-x", "--audio-format", "mp3", "--audio-quality", "5",
           "--download-sections", f"*0-{a.seconds}",
           "--postprocessor-args", f"ExtractAudio:-ac 1 -ar 22050 -t {a.seconds}",
           "-o", tmpl, f"https://www.youtube.com/watch?v={yt_id}"]
    if a.cookies_from_browser:
        cmd[3:3] = ["--cookies-from-browser", a.cookies_from_browser]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    ok = p.returncode == 0 and os.path.exists(os.path.join(out, f"{perf_id}.mp3"))
    err = (p.stderr or "").strip().splitlines()
    return ok, (err[-1] if err else f"returncode {p.returncode}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join("data", "shs-100k", "train.csv"))
    ap.add_argument("--out", default="audio")
    ap.add_argument("--n-works", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-work", type=int, default=5)
    ap.add_argument("--max-tries-per-work", type=int, default=15, help="give up on a work after this many attempts")
    ap.add_argument("--seconds", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--max-blocked", type=int, default=5, help="stop after this many bot-check errors in a row")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="OFF by default. e.g. 'firefox' or 'chrome'. Andy's choice; uses his browser's YouTube login.")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"): sys.exit("ffmpeg not found on PATH (see scripts/setup_windows.md)")
    os.makedirs(a.out, exist_ok=True)
    fail_csv, ok_csv = os.path.join(a.out, "failures.csv"), os.path.join(a.out, "downloaded.csv")
    failed = set() if a.retry_failed else read_set(fail_csv, "perf_id")
    have = lambda pid: os.path.exists(os.path.join(a.out, f"{pid}.mp3"))
    logged_ok = read_set(ok_csv, "perf_id")
    blocked_run, stop = 0, False
    sample = load_sample(a.csv, a.n_works, a.seed)
    for i, (work, rows) in enumerate(sample, 1):
        rows = rows[:]; random.Random(f"{a.seed}-{work}").shuffle(rows)
        n_ok = sum(have(r[0]) for r in rows); tries = 0
        for r in rows:
            if n_ok >= a.per_work or tries >= a.max_tries_per_work: break
            pid, yt = r[0], r[4]
            if have(pid) or pid in failed: continue
            tries += 1
            ok, msg = download(yt, pid, a.out, a)
            if ok:
                n_ok += 1; blocked_run = 0
                if pid not in logged_ok: append(ok_csv, ["perf_id", "work_id", "title", "artist", "yt_id"], r[:5])
            else:
                append(fail_csv, ["perf_id", "work_id", "yt_id", "error", "time"], [pid, work, yt, msg, time.strftime("%Y-%m-%d %H:%M:%S")])
                if any(m in msg.lower() for m in BLOCK_MARKERS):
                    blocked_run += 1
                    if blocked_run >= a.max_blocked:
                        print(f"STOP: YouTube bot check / rate limit {blocked_run}x in a row: {msg}"); stop = True; break
                else: blocked_run = 0
            time.sleep(a.sleep + random.uniform(0, 2))
        print(f"[{i}/{len(sample)}] work {work}: {n_ok} versions", flush=True)
        if stop: break
    per = defaultdict(int)
    for w, rows in sample:
        per[w] = sum(have(r[0]) for r in rows)
    print(f"\nworks with >=2 versions: {sum(v >= 2 for v in per.values())} / {len(sample)}"
          f"  (files: {sum(per.values())}; bar is >=200 works)")

if __name__ == "__main__":
    main()
