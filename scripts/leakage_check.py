#!/usr/bin/env python3
"""Find Covers80 songs that also appear as SHS-100K-2025 training works (normalised fuzzy title match,
plus artist overlap as supporting evidence). Writes the SHS work_ids to exclude from training.
  python scripts\\leakage_check.py --lists data\\covers80\\list1.txt data\\covers80\\list2.txt
Covers80 list lines look like: Song_Name/artist_name+Album_Name+01-Song_Name
Outputs: data/leakage_matches.csv (for manual review) and data/exclude_works.txt (one work_id per line).
Review the matches by hand: the title threshold is a heuristic."""
import argparse, csv, os, re, unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("_", " ").replace("&", " and ")
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)            # (live), [remastered] ...
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return " ".join(s.split())

def parse_list(path):
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            song, rest = line.split("/", 1)
            artist = rest.split("+")[0]
            out.append((song.replace("_", " "), artist.replace("_", " "), line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--shs", nargs="+", default=[os.path.join("data", "shs-100k", f) for f in ("train.csv",)])
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()
    c80 = [x for p in a.lists for x in parse_list(p)]
    titles = defaultdict(set); artists = defaultdict(set)
    for p in a.shs:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                titles[norm(r[2])].add(r[1]); artists[r[1]].add(norm(r[3]))
    keys = list(titles); rows, exclude = [], set()
    for song, artist, raw in c80:
        ns, na = norm(song), norm(artist)
        for k in keys:
            if not k or abs(len(k) - len(ns)) > max(4, len(ns) // 3): continue
            sc = 1.0 if k == ns else SequenceMatcher(None, ns, k).ratio()
            if sc < a.threshold: continue
            for w in titles[k]:
                art = max((SequenceMatcher(None, na, x).ratio() for x in artists[w]), default=0)
                rows.append([raw, song, artist, w, k, round(sc, 3), round(art, 3)]); exclude.add(w)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "leakage_matches.csv"), "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f); cw.writerow(["covers80_entry", "c80_title", "c80_artist", "shs_work_id", "shs_title_norm", "title_score", "best_artist_score"]); cw.writerows(rows)
    with open(os.path.join(a.out_dir, "exclude_works.txt"), "w") as f:
        f.write("\n".join(sorted(exclude, key=int)) + ("\n" if exclude else ""))
    songs = {r[1] for r in rows}
    print(f"{len(c80)} Covers80 entries; {len(songs)} distinct songs matched; {len(exclude)} SHS works to exclude")

if __name__ == "__main__":
    main()
