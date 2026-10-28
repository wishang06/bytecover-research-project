from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
REPO = DATA / "repo"
NR = "not reported"

PAPER_COVERS80_MAP = 0.906
PAPER_FULL_SONGS = 8858
PAPER_FULL_RECORDINGS = 108523
CHANCE_MAP = 0.036
SIZES = [50, 150, 300]
ARMS = {"A": "Arm A: CE only", "B": "Arm B: CE + triplet (m=0.3)"}
COLORS = {"A": "#1f77b4", "B": "#d62728"}

CHECKPOINT_TIMES_UTC = {
    "B": ["2026-10-25T05:15:51Z", "05:20:25", "05:21:15", "05:22:06", "05:22:57", "05:23:47", "05:26:14", "05:28:44", "05:31:11", "05:33:40", "05:36:08", "05:41:00", "05:45:53", "05:50:50", "05:55:44", "06:00:41"],
    "A": ["2026-10-25T20:07:13Z", "20:11:07", "20:11:32", "20:11:56", "20:12:21", "20:12:46", "20:13:59", "20:15:11", "20:16:23", "20:17:36", "20:18:48", "20:21:12", "20:23:34", "20:25:57", "20:28:19", "20:30:41"],
}


def recordings_per_size():
    m = pd.read_csv(REPO / "train_manifest.csv")
    per_work = m.groupby("work_id").size()
    out = {}
    for s in SIZES:
        lo = per_work.sort_values().iloc[:s].sum() if s == len(per_work) else int(per_work.min() + (s - 1) * per_work.max())
        hi = int(s * per_work.max()) if s < len(per_work) else int(per_work.sum())
        out[s] = (int(min(lo, hi)), hi)
    return out, len(m), int(m["excluded"].sum())


def run_minutes():
    out = {}
    for arm, stamps in CHECKPOINT_TIMES_UTC.items():
        day = stamps[0][:11]
        t = pd.to_datetime([stamps[0]] + [f"{day}{s}Z" for s in stamps[1:]])
        gaps = np.diff(t.values).astype("timedelta64[s]").astype(float) / 60
        out[arm] = {s: gaps[i * 5 : i * 5 + 5] for i, s in enumerate(SIZES)}
        out[arm]["total"] = gaps.sum()
    return out


def t_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h, n


def summarise(runs):
    rows = []
    for arm in ARMS:
        for size in SIZES:
            m, lo, hi, n = t_ci(runs.query("arm == @arm and size == @size")["map"])
            rows.append(dict(arm=arm, size=size, mean=m, lo=lo, hi=hi, n=n))
    return pd.DataFrame(rows)


def paired(runs):
    wide = runs.pivot_table(index=["size", "seed"], columns="arm", values="map").reset_index()
    rows = []
    for size in SIZES:
        d = wide.query("size == @size")
        m, lo, hi, n = t_ci(d["B"] - d["A"])
        rows.append(dict(size=size, mean=m, lo=lo, hi=hi, n=n))
    return pd.DataFrame(rows)


def fmt_ci(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_rec(r):
    return f"{r[1]:,}" if r[0] == r[1] else f"{r[0]:,}-{r[1]:,}"


def build_tables(runs, summary, diffs, recs, n_rec, mins):
    epochs = sorted(runs["epochs"].unique())
    gpus = ", ".join(sorted(runs["gpu"].unique()))
    n_excl = len([l for l in (REPO / "exclude_works.txt").read_text().splitlines() if l.strip()])
    cols = ["Item", "ByteCover paper (arXiv:2010.14022v2)", "Paper source", "Andy's recreation", "Recreation source"]

    setup = [
        ["Backbone", "ResNet-IBN 50 + GeM + BNNeck", "p.2 Fig.1, §2.1; p.3 §2.2", "ResNet-50-IBN (IBN in stages 1-3) + GeM + BNNeck", "model.py, loss.py"],
        ["Loss", "CE(Softmax(W f_c)) + triplet on f_t", "p.3 Eq.(2)-(3)", "A: CE on f_c only; B: CE + batch-hard triplet on f_t", "loss.py"],
        ["Triplet margin", "0.3", "p.3 §2.2", "0.3 (Euclidean, batch-hard)", "loss.py"],
        ["Optimiser", "Adam", "p.3 §2.2", "Adam, weight decay 0, no schedule", "train.py"],
        ["Learning rate", "0.0004", "p.3 §2.2", "0.0004", "train.py"],
        ["Batch size", "32", "p.3 §2.2", "30 clips (10 works x 3)", "train.py"],
        ["Epochs", NR, "-", f"{', '.join(map(str, epochs))}, no early stopping", "runs.csv, train.py"],
        ["Optimizer steps", NR, "-", "500 / 1,500 / 3,000 (50/150/300 songs)", "PLAN.md"],
        ["Embedding dim", "2048 (128/256 via FC head also tested)", "p.3 §2.1; p.4 Table 2", "2048", "loss.py"],
        ["CQT implementation", NR, "-", "nnAudio CQT2010v2", "cache_cqt.py"],
        ["CQT sample rate", "22,050 Hz", "p.2 §2.1", "22,050 Hz", "cache_cqt.py"],
        ["CQT bins", "12 / octave; 84 total", "p.2 §2.1", "12 / octave; 84 total", "cache_cqt.py"],
        ["CQT window / hop", "Hann / 512", "p.2 §2.1", "Hann / 512", "cache_cqt.py"],
        ["CQT time averaging", "factor 100", "p.2 §2.1", "factor 100 (--compress 100)", "PLAN.md, dataset.py"],
        ["Audio length used", NR, "-", "first 200 s (padded to 86 frames)", "README.md, dataset.py"],
        ["Tempo augmentation", NR, "-", "none (CQTs cached)", "README.md"],
    ]
    data = [
        ["Training source", "SHS100K training subset", "p.3-4 §3.1", "SHS-100K-2025 works (YouTube audio)", "README.md"],
        ["SHS100K full (all splits)", f"{PAPER_FULL_SONGS:,} songs / {PAPER_FULL_RECORDINGS:,} rec.", "p.3 §3.1", "-", "-"],
        ["Split ratio", "train:val:test = 8:1:1", "p.3 §3.1", "-", "-"],
        ["Training songs", f"{NR} (train-split count)", "-", "50 / 150 / 300 (nested prefixes)", "dataset.py"],
        ["Training recordings", f"{NR} (train-split count)", "-", f"{fmt_rec(recs[50])} / {fmt_rec(recs[150])} / {fmt_rec(recs[300])}", "train_manifest.csv"],
        ["Recordings per song", NR, "-", "5 (299 works), 3 (1 work)", "train_manifest.csv"],
        ["Covers80 leakage removal", NR, "-", f"{n_excl} works excluded", "exclude_works.txt"],
        ["Test set", "Covers80: 80 songs x 2 = 160 rec., all pairs", "p.4 §3.1", "Covers80, 160 recordings", "README.md"],
        ["Similarity", "cosine distance", "p.4 §3.1", "cosine on f_c", "loss.py"],
    ]
    a_tot, b_tot = mins["A"]["total"], mins["B"]["total"]
    per = lambda arm: " / ".join(f"{mins[arm][s].mean():.1f}" for s in SIZES)
    compute = [
        ["Hardware", NR, "-", f"Google Colab, {gpus} (all 30 runs)", "runs.csv, train_loss.csv"],
        ["Epoch time", NR, "-", "0.18 min/epoch (size 300, timing run)", "PLAN.md"],
        ["Min per run, arm A (50/150/300)", NR, "-", per("A"), "Drive checkpoint times*"],
        ["Min per run, arm B (50/150/300)", NR, "-", per("B"), "Drive checkpoint times*"],
        ["Total wall clock, 30 runs", NR, "-", f"~{a_tot + b_tot:.0f} min (A {a_tot:.0f}, B {b_tot:.0f})", "Drive checkpoint times*"],
        ["Colab compute units used", NR, "-", "TBD (from Colab logs)", "not found in repo or Drive"],
        ["Cost", NR, "-", "Colab Pro USD 9.99 + 0.80 tax / month (subscription)", "Drive: Spending Tracker"],
        ["Training runs", NR, "-", "30 (2 arms x 3 sizes x 5 seeds)", "runs.csv"],
    ]
    results = [
        ["Covers80 mAP", f"{PAPER_COVERS80_MAP:.3f}", "p.4 Table 1", "0.033-0.079 (see per-size table)", "runs.csv"],
        ["Covers80 P@10", "0.093", "p.4 Table 1", "not in provided data", "-"],
        ["Covers80 MR1", "3.54", "p.4 Table 1", "not in provided data", "-"],
        ["Chance mAP (Covers80)", NR, "-", f"~{CHANCE_MAP}", "brief"],
    ]
    per_size_cols = ["Songs", "Recordings", "Arm A mAP [95% t-CI]", "Arm B mAP [95% t-CI]", "B - A paired [95% t-CI]", "Paper mAP"]
    per_size = []
    for s in SIZES:
        a = summary.query("arm == 'A' and size == @s").iloc[0]
        b = summary.query("arm == 'B' and size == @s").iloc[0]
        d = diffs.query("size == @s").iloc[0]
        per_size.append([str(s), fmt_rec(recs[s]), fmt_ci(a["mean"], a.lo, a.hi), fmt_ci(b["mean"], b.lo, b.hi), fmt_ci(d["mean"], d.lo, d.hi), f"{PAPER_COVERS80_MAP:.3f}"])

    return [
        ("Setup", cols, setup),
        ("Data", cols, data),
        ("Compute", cols, compute),
        ("Results", cols, results),
        ("Covers80 mAP by training size (n=5 seeds, df=4)", per_size_cols, per_size),
    ]


NOTES = [
    "* Minutes per run = gap between consecutive checkpoint saves in Drive (includes Covers80 eval and save); first run timed from the results folder/runs.csv creation time.",
    "Recordings at 50/150 songs are a range: 299 works have 5 recordings, 1 has 3, and data/work_order.txt (not in the repo bundle) decides which prefix holds it.",
]


def to_markdown(title, cols, rows):
    lines = [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def is_missing(text):
    return text.startswith(NR) or text.startswith("TBD") or text.startswith("not in provided")


def table_png(title, cols, rows, path, note=None):
    widths = [len(c) * 1.2 for c in cols]
    for r in rows:
        widths = [max(w, len(v)) + 2 for w, v in zip(widths, r)]
    total = sum(widths)
    fig_w = total * 0.085
    fig_h = 0.36 * (len(rows) + 1) + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, colWidths=[w / total for w in widths], cellLoc="left", loc="upper center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1, 1.4)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#2f3e4e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f4f6f8" if r % 2 else "white")
            if is_missing(cell.get_text().get_text()):
                cell.get_text().set_color("#9a4d00")
                cell.get_text().set_style("italic")
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=6)
    if note:
        fig.text(0.01, 0.0, note, fontsize=8, color="#555555", ha="left", va="top")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_training_data(recs, path):
    labels = ["ByteCover paper\nSHS100K, all splits\n(train split not reported)"] + [f"Recreation\n{s} songs" for s in SIZES]
    songs = [PAPER_FULL_SONGS] + SIZES
    rec_hi = [PAPER_FULL_RECORDINGS] + [recs[s][1] for s in SIZES]
    rec_lbl = [f"{PAPER_FULL_RECORDINGS:,}"] + [fmt_rec(recs[s]) for s in SIZES]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w / 2, songs, w, label="Songs (works)", color="#4c72b0")
    b2 = ax.bar(x + w / 2, rec_hi, w, label="Recordings", color="#dd8452")
    ax.set_yscale("log")
    ax.set_ylim(10, 4e5)
    for rect, v in zip(b1, songs):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.12, f"{v:,}", ha="center", fontsize=9)
    for rect, v, lbl in zip(b2, rec_hi, rec_lbl):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.12, lbl, ha="center", fontsize=9)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Training data: ByteCover paper vs recreation")
    ax.legend(loc="upper right")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.text(0.0, -0.24, "Paper bars: full SHS100K as stated (p.3 §3.1); split 8:1:1, train-split counts not reported.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_map(summary, runs, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    offsets = {"A": 0.96, "B": 1.04}
    for arm, label in ARMS.items():
        s = summary.query("arm == @arm").sort_values("size")
        xs = s["size"].to_numpy() * offsets[arm]
        ax.errorbar(xs, s["mean"], yerr=[s["mean"] - s.lo, s.hi - s["mean"]], fmt="o-", capsize=4, color=COLORS[arm], label=f"{label}, mean ± 95% t-CI")
        pts = runs.query("arm == @arm")
        ax.scatter(pts["size"] * offsets[arm], pts["map"], s=12, color=COLORS[arm], alpha=0.35, zorder=1)
    ax.axhline(PAPER_COVERS80_MAP, color="black", ls="--", lw=1.2, label=f"ByteCover paper: {PAPER_COVERS80_MAP} (Table 1, SHS100K train)")
    ax.axhline(CHANCE_MAP, color="grey", ls=":", lw=1.4, label=f"Chance ~ {CHANCE_MAP}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SIZES, [str(s) for s in SIZES])
    ax.minorticks_off()
    ax.set_xlim(38, 400)
    ax.set_ylim(0.02, 1.2)
    yt = [0.025, 0.036, 0.05, 0.1, 0.2, 0.5, 0.906]
    ax.set_yticks(yt, [str(v) for v in yt])
    ax.set_xlabel("Training songs (log scale)")
    ax.set_ylabel("Covers80 mAP (log scale)")
    ax.set_title("Covers80 mAP vs training size (5 seeds per point)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="center right")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    runs = pd.read_csv(REPO / "runs.csv")
    recs, n_rec, _ = recordings_per_size()
    mins = run_minutes()
    summary = summarise(runs)
    diffs = paired(runs)
    tables = build_tables(runs, summary, diffs, recs, n_rec, mins)

    md = ["# ByteCover (Du et al., arXiv:2010.14022v2) vs Andy's small-scale recreation", ""]
    for i, (title, cols, rows) in enumerate(tables, 1):
        md.append(to_markdown(title, cols, rows))
        png = f"table_{i}_{title.split(' ')[0].lower()}.png"
        note = {"Compute": NOTES[0], "Data": NOTES[1]}.get(title)
        if note:
            md.append(note + "\n")
        table_png(title, cols, rows, HERE / png, note)
        md.append(f"![{title}]({png})\n")
    md.append("## Plots\n\n![Training data](training_data.png)\n\n![Covers80 mAP vs size](map_vs_size.png)\n")
    (HERE / "comparison.md").write_text("\n".join(md))

    plot_training_data(recs, HERE / "training_data.png")
    plot_map(summary, runs, HERE / "map_vs_size.png")
    summary.to_csv(HERE / "map_summary.csv", index=False)
    diffs.to_csv(HERE / "paired_diff_summary.csv", index=False)


if __name__ == "__main__":
    main()
