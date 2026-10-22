#!/usr/bin/env python3
"""Download + extract covers80.tgz (~164 MB) from LabROSA (Columbia). Plain Python, Windows-safe.
  python scripts\\get_covers80.py --out data\\covers80
Result: data/covers80/coversongs/covers32k/<Song>/<file>.mp3 and list1.txt / list2.txt."""
import argparse, os, shutil, sys, tarfile, urllib.request

BASE = "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/"

def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("exists:", dest); return
    print("downloading", url)
    with urllib.request.urlopen(url, timeout=60) as r, open(dest + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if total: print(f"\r{done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    print(); os.replace(dest + ".part", dest)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join("data", "covers80"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for name in ("list1.txt", "list2.txt"):
        fetch(BASE + name, os.path.join(a.out, name))
    tgz = os.path.join(a.out, "covers80.tgz"); fetch(BASE + "covers80.tgz", tgz)
    with tarfile.open(tgz) as t:
        for m in t.getmembers():  # refuse path traversal
            p = os.path.realpath(os.path.join(a.out, m.name))
            if not p.startswith(os.path.realpath(a.out)): sys.exit(f"unsafe path in tar: {m.name}")
        t.extractall(a.out)
    n = sum(f.endswith(".mp3") for _, _, fs in os.walk(a.out) for f in fs)
    print(f"extracted {n} mp3 files into {a.out} (expected 160)")

if __name__ == "__main__":
    main()
