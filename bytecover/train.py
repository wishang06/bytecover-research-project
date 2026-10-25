"""One training run. Both arms call this with the same size and seed.

The seed fixes the encoder's initial weights and the batch order. The loss
function is the only intended difference between arms. Covers80 is not read
here, and there is no early stopping: the weights after `epochs` are the run.

Batches are 10 works by 3 recordings (30 clips). Every work in the size is in
exactly one batch per epoch. The paper uses 32, which does not divide 50, 150,
and 300. Adam uses learning rate 0.0004 and weight decay 0, matching the
paper's stated optimizer with no extra schedule.

`loss_fn(f_t, y)` is Arm A or Arm B from bytecover.loss. If it is a module, its
parameters are trained with the encoder. The return value is (encoder, loss_fn).
Every epoch is appended to the loss log. The 30-run grid uses EPOCHS (100) and require_l4=True.
"""
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bytecover.dataset import CQTClips, collate_clips, subset_work_ids
from bytecover.model import ResNet50IBN

WORKS_PER_BATCH = 10
CLIPS_PER_WORK = 3
LEARNING_RATE = 4e-4
EPOCHS = 100
LOSS_COLUMNS = ["arm", "size", "seed", "epoch", "mean_objective", "gpu"]


class PKSampler:
    """Yields index lists: 10 works, 3 clips each, every work once per epoch."""

    def __init__(self, dataset, work_ids, generator):
        self.generator = generator
        self.work_ids = list(work_ids)
        self.groups = {work_id: [] for work_id in self.work_ids}
        for index, row in enumerate(dataset.rows):
            self.groups[row["work_id"]].append(index)
        short = [work_id for work_id, idxs in self.groups.items() if len(idxs) < CLIPS_PER_WORK]
        if short or len(self.work_ids) % WORKS_PER_BATCH != 0:
            raise ValueError(f"need groups of {WORKS_PER_BATCH} works with at least {CLIPS_PER_WORK} clips, short={short[:5]}")

    def __len__(self):
        return len(self.work_ids) // WORKS_PER_BATCH

    def __iter__(self):
        order = torch.randperm(len(self.work_ids), generator=self.generator).tolist()
        works = [self.work_ids[i] for i in order]
        for start in range(0, len(works), WORKS_PER_BATCH):
            chosen = []
            for work_id in works[start : start + WORKS_PER_BATCH]:
                pool = self.groups[work_id]
                pick = torch.randperm(len(pool), generator=self.generator)[:CLIPS_PER_WORK].tolist()
                chosen.extend(pool[i] for i in pick)
            yield chosen


def make_encoder(seed):
    torch.manual_seed(seed)
    return ResNet50IBN()


def gpu_name(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def check_l4(name):
    """The 30-run grid counts only on a Colab L4."""
    if "L4" not in name:
        raise RuntimeError(f"this run requires a Colab L4, got {name}")


def append_loss_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOSS_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def train(loss_fn, rows, cqt_dir, work_order, size, seed, epochs, arm, device="cpu", log_path="results/train_loss.csv", require_l4=False):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    work_ids = subset_work_ids(work_order, size)
    dataset = CQTClips(rows, cqt_dir, work_ids)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = PKSampler(dataset, work_ids, generator)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_clips)
    name = gpu_name(device)
    if require_l4:
        check_l4(name)
    model = make_encoder(seed).to(device)
    params = list(model.parameters())
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
        loss_fn.train()
        params += list(loss_fn.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=0)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        steps = 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            length = batch["length"].to(device)
            optimizer.zero_grad(set_to_none=True)
            f_t = model(x, length)
            loss = loss_fn(f_t, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite objective at epoch {epoch + 1}, step {steps + 1}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        mean_objective = total / steps
        print(f"arm {arm} size {size} seed {seed} epoch {epoch + 1}/{epochs} mean_objective {mean_objective:.6f} gpu {name}", flush=True)
        append_loss_row(log_path, {
            "arm": arm,
            "size": size,
            "seed": seed,
            "epoch": epoch + 1,
            "mean_objective": f"{mean_objective:.6f}",
            "gpu": name,
        })
    model.eval()
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn.eval()
    return model, loss_fn
