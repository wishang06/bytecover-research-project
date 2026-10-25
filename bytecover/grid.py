"""The 30-run grid. Covers80 is scored only after a run has finished training.

Order is size 50, then 150, then 300. Inside a size, seeds 0 to 4, and for
each seed Arm A then Arm B. A finished run is a row in runs.csv. Starting the
grid again skips those rows. A checkpoint with no row is scored without
training again. A half-written loss log for an unfinished run is removed
before that run starts over.

Outputs go to out_dir, which on Colab should be the Drive folder so a
disconnect does not delete them. The paired t-interval table is not written
here.
"""
import csv
from pathlib import Path

import torch

from bytecover.dataset import load_manifest, load_work_order
from bytecover.evaluate import cover_pairs, embed_paths, mean_average_precision
from bytecover.loss import make_loss
from bytecover.model import ResNet50IBN
from bytecover.train import EPOCHS, LOSS_COLUMNS, check_l4, gpu_name, train

SIZES = (50, 150, 300)
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("A", "B")
RUN_COLUMNS = ["arm", "size", "seed", "map", "gpu", "epochs"]


def run_tag(arm, size, seed):
    return f"{arm}_s{size}_k{seed}"


def read_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def is_done(runs_path, arm, size, seed):
    for row in read_rows(runs_path):
        if row["arm"] == arm and int(row["size"]) == size and int(row["seed"]) == seed:
            return True
    return False


def drop_loss_rows(path, arm, size, seed):
    rows = [
        row for row in read_rows(path)
        if not (row["arm"] == arm and int(row["size"]) == size and int(row["seed"]) == seed)
    ]
    if Path(path).exists():
        write_rows(path, LOSS_COLUMNS, rows)


def append_run(path, row):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def load_checkpoint(path, arm, size, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    encoder = ResNet50IBN().to(device)
    head = make_loss(arm, size).to(device)
    encoder.load_state_dict(saved["encoder"])
    head.load_state_dict(saved["head"])
    encoder.eval()
    head.eval()
    return encoder, head


def score_run(encoder, head, paths, song_id, device):
    embeddings = embed_paths(encoder, head, paths, device=device)
    return mean_average_precision(embeddings, song_id)


def run_grid(manifest, work_order, cqt_shs, covers_cqt, list1, list2, out_dir, device="cuda", epochs=EPOCHS):
    name = gpu_name(device)
    check_l4(name)
    out = Path(out_dir)
    runs_path = out / "runs.csv"
    loss_path = out / "train_loss.csv"
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths, song_id = cover_pairs(list1, list2, covers_cqt)
    rows = load_manifest(manifest)
    order = load_work_order(work_order)
    jobs = [(size, seed, arm) for size in SIZES for seed in SEEDS for arm in ARMS]
    done = sum(is_done(runs_path, arm, size, seed) for size, seed, arm in jobs)
    print(f"gpu {name}", flush=True)
    print(f"{len(jobs)} runs, {epochs} epochs, {done} already finished, output {out}", flush=True)
    for number, (size, seed, arm) in enumerate(jobs, start=1):
        tag = run_tag(arm, size, seed)
        if is_done(runs_path, arm, size, seed):
            print(f"[{number}/{len(jobs)}] skip {tag}", flush=True)
            continue
        ckpt = ckpt_dir / f"{tag}.pt"
        print(f"[{number}/{len(jobs)}] {tag}", flush=True)
        if ckpt.exists():
            encoder, head = load_checkpoint(ckpt, arm, size, device)
        else:
            drop_loss_rows(loss_path, arm, size, seed)
            encoder, head = train(
                make_loss(arm, size), rows, cqt_shs, order,
                size=size, seed=seed, epochs=epochs, arm=arm,
                device=device, log_path=loss_path, require_l4=True,
            )
            torch.save({
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "arm": arm,
                "size": size,
                "seed": seed,
                "gpu": name,
                "epochs": epochs,
            }, ckpt)
        value = score_run(encoder, head, paths, song_id, device)
        append_run(runs_path, {
            "arm": arm,
            "size": size,
            "seed": seed,
            "map": f"{value:.6f}",
            "gpu": name,
            "epochs": epochs,
        })
        print(f"[{number}/{len(jobs)}] {tag} mAP {value:.4f}", flush=True)
        del encoder, head
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"finished {runs_path}", flush=True)
