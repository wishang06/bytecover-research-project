"""Covers80 mean average precision. This is the final test, not a training signal.

list1.txt line i and list2.txt line i are the two versions of song i. All 160
tracks are queries. For each query the other 159 are ranked by cosine
similarity of the arm's retrieval vector (f_t for Arm A, f_c for Arm B). Each
query has one relevant track, so its average precision is 1 over the rank of
that partner. mAP is the mean of those 160 scores.

The four extra files in cqt/covers80 that are not on these lists are not scored.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bytecover.dataset import N_BINS, PAD_FRAMES


def read_list(path):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def npy_name(list_line):
    rel = list_line.replace("/", "__").replace("\\", "__")
    return f"coversongs__covers32k__{rel}.npy"


def cover_pairs(list1, list2, cqt_dir):
    """Return paths and song ids, two entries per song, list1 then list2."""
    left = read_list(list1)
    right = read_list(list2)
    if len(left) != len(right) or len(left) == 0:
        raise ValueError(f"list lengths {len(left)} and {len(right)}")
    paths = []
    song_id = []
    root = Path(cqt_dir)
    for i, (a, b) in enumerate(zip(left, right)):
        for line in (a, b):
            path = root / npy_name(line)
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
            song_id.append(i)
    return paths, song_id


def average_precision(is_relevant):
    """is_relevant is the ranked list, best first. One number per query."""
    hits = 0
    total = 0.0
    for rank, relevant in enumerate(is_relevant, start=1):
        if relevant:
            hits += 1
            total += hits / rank
    if hits == 0:
        return 0.0
    return total / hits


def mean_average_precision(embeddings, song_id):
    """embeddings is (N, D). song_id[i] == song_id[j] means i and j are covers."""
    if len(embeddings) != len(song_id):
        raise ValueError("one song id per embedding")
    z = F.normalize(embeddings.float(), dim=1)
    similarity = z @ z.T
    similarity.fill_diagonal_(float("-inf"))
    order = similarity.argsort(dim=1, descending=True)
    songs = list(song_id)
    scores = []
    for i in range(len(songs)):
        ranked = [songs[j] == songs[i] for j in order[i].tolist() if j != i]
        scores.append(average_precision(ranked))
    return float(sum(scores) / len(scores))


def embed_paths(encoder, head, paths, device="cpu", batch_size=16):
    encoder.eval()
    head.eval()
    vectors = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images = []
        lengths = []
        for path in chunk:
            array = np.load(path)
            if array.ndim != 2 or array.shape[0] != N_BINS:
                raise ValueError(f"{path} has shape {array.shape}")
            length = int(array.shape[1])
            if length < 1 or length > PAD_FRAMES:
                raise ValueError(f"{path} width {length} is outside 1..{PAD_FRAMES}")
            image = np.zeros((1, N_BINS, PAD_FRAMES), dtype=np.float32)
            image[0, :, :length] = array.astype(np.float32)
            images.append(image)
            lengths.append(length)
        x = torch.from_numpy(np.stack(images)).to(device)
        length = torch.tensor(lengths, device=device)
        with torch.no_grad():
            f_t = encoder(x, length)
            vectors.append(head.embed(f_t).detach().cpu())
    return torch.cat(vectors, dim=0)
