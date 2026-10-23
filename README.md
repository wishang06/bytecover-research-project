# bytecover-research-project

A small-scale replication study of **ByteCover** (Du et al., ByteDance, ICASSP 2021), a deep learning model for cover song identification (CSI). CSI means recognising that two recordings are versions of the same song, even when they differ in key, tempo, instruments or structure.

The model and method are ByteDance's. This project reproduces them at a smaller scale, and the data-size comparison is my own study. It follows on from a paper I wrote at Fudan University explaining the ByteCover approach.

> Status: in progress. Data and features are prepared, and training has not started yet. No results are reported here.

## Research question

ByteCover trains a ResNet-IBN network with two losses at once: a **classification loss** (which song is this?) and a **triplet loss** (put versions of the same song close together and different songs far apart), joined through a **BNNeck** layer. The paper shows this combination beats classification alone when trained on the full SHS-100K dataset.

**Does the combined loss still help when there is much less training data?**

## Experiment design

| | |
|---|---|
| Arm A | Classification loss only (the paper's "+IBN" configuration) |
| Arm B | Classification + triplet loss with BNNeck (full ByteCover) |
| Training data | SHS-100K-2025: 300 works sampled with seed 0, 13 removed for test-set overlap, 287 kept |
| Training sizes | 50, 150 and 287 works, nested (each smaller set is a subset of the larger one) |
| Test set | Covers80 (80 songs, 2 versions each, 160 recordings) |
| Seeds | 5 paired seeds (0 to 4), each used for both arms |
| Metric | Mean average precision (mAP) on Covers80 |
| Analysis | Per-seed difference B minus A at each size, with a 95% t-interval |

That's 3 sizes, 2 arms and 5 seeds, so 30 training runs.

**Rules fixed before any results:** both arms train for the same fixed number of epochs with no early stopping. Covers80 is used only for the final test, never for tuning. The analysis plan is set in advance.

## Pipeline

1. **Download audio** (`scripts/download_audio.py`). SHS-100K is a list of YouTube IDs, so audio is fetched with `yt-dlp`. `ffmpeg` keeps the first 200 s and converts it to mono at 22,050 Hz. The script takes up to 5 versions per work and can resume after interruptions. Result: 1,498 recordings, and all 300 works have at least 2 versions.
2. **Download the test set** (`scripts/get_covers80.py`). This fetches Covers80 from Columbia's LabROSA site.
3. **Leakage check** (`scripts/leakage_check.py`). It fuzzy-matches song titles and artists between Covers80 and the training works, so no test song is seen in training. Matches were reviewed by hand, and all 13 flagged works are excluded (`data/exclude_works.txt`).
4. **Feature caching** (`scripts/cache_cqt.py`). It computes a constant-Q transform (CQT) with nnAudio: 84 bins at 12 per octave (one per semitone over 7 octaves), hop length 512, averaged over every 100 frames as in the paper. The features are saved as float16 `.npy` files, so training never recomputes them.
5. **Training and evaluation** (`notebooks/ablation.ipynb`) runs on a Google Colab GPU.

## Results

Pending.

## Known differences from the paper

- **Scale.** We use 287 works, while the paper uses the full SHS-100K.
- **Time augmentation.** The CQTs are cached ahead of time, so tempo augmentation on the raw complex CQT isn't available.
- **Data.** We use SHS-100K-2025, and some original YouTube links are no longer available.

## Setup (Windows)

```powershell
winget install -e --id astral-sh.uv
winget install -e --id Gyan.FFmpeg
uv sync
git clone --depth 1 https://github.com/second-hand-songs/shs-100k data\shs-100k
uv run python scripts\download_audio.py --csv data\shs-100k\train.csv --out audio
uv run python scripts\get_covers80.py --out data\covers80
uv run python scripts\leakage_check.py --lists data\covers80\list1.txt data\covers80\list2.txt
uv run python scripts\cache_cqt.py --audio audio --out cqt\shs --compress 100
uv run python scripts\cache_cqt.py --audio data\covers80 --out cqt\covers80 --compress 100
```

`audio/`, `cqt/` and `data/shs-100k/` are kept out of Git. See `PLAN.md` for step-by-step progress.

## References

- Du, X., Yu, Z., Zhu, B., Chen, X., Ma, Z. (2021). *ByteCover: Cover Song Identification via Multi-Loss Training.* ICASSP 2021. [arXiv:2010.14022](https://arxiv.org/abs/2010.14022)
- Orfium's open-source ByteCover reimplementation (used as a reference for CQT settings).
- SHS-100K (Xu et al., 2018) and Covers80 (Ellis, 2007).

## Author

Wang Ieong Shang, University of Pennsylvania.

