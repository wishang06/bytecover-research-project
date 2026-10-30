# bytecover-research-project

A small-scale replication study of **ByteCover** (Du et al., ByteDance, ICASSP 2021), a deep learning model for cover song identification (CSI). CSI means recognising that two recordings are versions of the same song, even when they differ in key, tempo, instruments or structure.

The model and method are ByteDance's. This project reproduces them at a smaller scale, and the data-size comparison is my own study. It follows on from a paper I wrote at Fudan University explaining the ByteCover approach.

> Status: the 30-run grid is complete. The comparison below uses the corrected Arm A, which has the same BNNeck and retrieval vector as Arm B.

## Research question

ByteCover trains a ResNet-IBN network with two losses at once: a **classification loss** (which song is this?) and a **triplet loss** (put versions of the same song close together and different songs far apart), joined through a **BNNeck** layer. The paper shows this combination beats classification alone when trained on the full SHS-100K dataset.

**Does the combined loss still help when there is much less training data?**

## Experiment design

| | |
|---|---|
| Arm A | Classification only. Same encoder, BNNeck, and bias-free classifier as Arm B. Retrieval uses f_c |
| Arm B | Arm A plus a batch-hard triplet on f_t. Retrieval uses the same f_c |
| Training data | SHS-100K-2025: 300 works sampled with seed 0. A leakage audit of the full train split flagged 13 other works; none of those 13 are in this sample, so all 300 are kept |
| Training sizes | 50, 150 and 300 works, nested (each smaller set is a subset of the larger one) |
| Test set | Covers80 (80 songs, 2 versions each, 160 recordings) |
| Seeds | 5 paired seeds (0 to 4), each used for both arms |
| Metric | Mean average precision (mAP) on Covers80 |
| Analysis | Per-seed difference B minus A at each size, with a 95% t-interval |

That's 3 sizes, 2 arms and 5 seeds, so 30 training runs.

**Rules fixed before any results:** both arms train for 100 epochs with no early stopping. Covers80 is used only for the final test, never for tuning. The analysis plan is set in advance. Both arms share the encoder, including GeM, so the comparison is the loss. The paper's "+IBN" row still used average pooling; copying that row would have mixed a pooling change into the loss comparison. Each batch is 10 works and 3 recordings (30 clips), so every work is seen once per epoch and a triplet has a positive pair in the batch. The paper's batch of 32 does not divide 50, 150, and 300.

## Pipeline

1. **Download audio** (`scripts/download_audio.py`). SHS-100K is a list of YouTube IDs, so audio is fetched with `yt-dlp`. `ffmpeg` keeps the first 200 s and converts it to mono at 22,050 Hz. The script takes up to 5 versions per work and can resume after interruptions. Result: 1,498 recordings, and all 300 works have at least 2 versions.
2. **Download the test set** (`scripts/get_covers80.py`). This fetches Covers80 from Columbia's LabROSA site.
3. **Leakage check** (`scripts/leakage_check.py`). It fuzzy-matches song titles and artists between Covers80 and the full SHS train split. Thirteen works match (`data/exclude_works.txt`). None of them are in the seed-0 sample, so the training pool keeps all 300 works. `scripts/make_manifest.py` writes `data/train_manifest.csv`, one row per recording.
4. **Feature caching** (`scripts/cache_cqt.py`). It computes a constant-Q transform (CQT) with nnAudio: 84 bins at 12 per octave (one per semitone over 7 octaves), hop length 512, averaged over every 100 frames as in the paper. The features are saved as float16 `.npy` files, so training never recomputes them.
5. **Training and evaluation** (`notebooks/ablation.ipynb`) runs on a Google Colab GPU.

## Diagrams

### Project pipeline

```mermaid
flowchart LR
    classDef done fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef todo fill:#fff3e0,stroke:#ef6c00,color:#e65100

    subgraph D["Data"]
        direction TB
        d1["SHS-100K-2025<br/>300 works · 1,498 recordings"]
        d2["Covers80<br/>160 recordings · test only"]
        d3["Leakage check<br/>13 flagged · 0 in sample"]
    end

    subgraph P["Features"]
        direction TB
        p1["CQT · nnAudio<br/>84 bins · hop 512"]
        p2["Compress ×100<br/>float16 .npy cache"]
    end

    subgraph M["Model"]
        direction TB
        m1["ResNet50-IBN"]
        m2["GeM pooling → f_t"]
        m3["BNNeck → f_c"]
    end

    subgraph E["Experiment"]
        direction TB
        e1["30 runs<br/>3 sizes × 2 arms × 5 seeds"]
        e2["Covers80 mAP<br/>cosine on f_c"]
        e3["Paired B − A<br/>95% t-interval"]
    end

    subgraph R["Remaining"]
        direction TB
        r1["Loss-curve check"]
        r2["Report"]
    end

    D --> P --> M --> E --> R
    class d1,d2,d3,p1,p2,m1,m2,m3,e1,e2,e3 done
    class r1,r2 todo
```

Green steps are finished and orange steps are still to do.

### Model and losses

```mermaid
flowchart LR
    x["CQT input"] --> enc["ResNet50-IBN"] --> gem["GeM pooling"] --> ft(["f_t · 2048-d"])
    ft --> bn["BNNeck<br/>BatchNorm1d"] --> fc(["f_c"])
    fc --> cls["Bias-free classifier"] --> ce["Cross-entropy<br/>Arms A and B"]
    ft --> tri["Batch-hard triplet<br/>margin 0.3<br/>Arm B only"]
    fc --> ret["Retrieval vector<br/>cosine similarity"]
```

Both arms share the encoder, pooling, BNNeck and classifier, so the only difference is the triplet loss on f_t.

### Training grid

```mermaid
flowchart TB
    subgraph S["Nested training sizes (songs)"]
        direction LR
        s50["50"] -->|subset of| s150["150"] -->|subset of| s300["300"]
    end
    S --> A["Arm A · cross-entropy"]
    S --> B["Arm B · cross-entropy + triplet"]
    A --> seeds["Seeds 0–4, paired across arms"]
    B --> seeds
    seeds --> runs["30 independent runs<br/>100 epochs each · no early stopping"]
```

| Songs | Batches per epoch | Optimizer steps (100 epochs) |
|---|---|---|
| 50 | 5 | 500 |
| 150 | 15 | 1,500 |
| 300 | 30 | 3,000 |

Every batch is 10 works × 3 clips (30 clips), trained with Adam at learning rate 0.0004 and weight decay 0. A larger set adds batches, not batch size, so the larger sizes also get more optimizer steps.

Detailed diagram sources: [docs/diagrams.md](docs/diagrams.md).

Report plan for office hours: [docs/report_outline.md](docs/report_outline.md).
## Results

Covers80 mAP after 100 epochs. Each cell is the mean over five paired seeds. Delta is mAP for Arm B minus mAP for Arm A. The interval is a 95% paired t-interval on those five differences (4 degrees of freedom).

| Works | Mean mAP A | Mean mAP B | Mean delta | 95% interval |
|---|---|---|---|---|
| 50 | 0.0330 | 0.0480 | 0.0150 | 0.0068 to 0.0233 |
| 150 | 0.0451 | 0.0558 | 0.0108 | −0.0120 to 0.0335 |
| 300 | 0.0562 | 0.0794 | 0.0232 | 0.0019 to 0.0445 |

The intervals at 50 and 300 works exclude zero. The interval at 150 works includes zero. With five seeds the intervals are wide. A random ranking of 159 other tracks would score about 0.035, so these absolute scores are close to that floor. An earlier Arm A, trained without the BNNeck, was discarded. Those scores are in `results/runs_discarded_biased_head.csv`.

## Known differences from the paper

- **Scale.** We use 300 works, while the paper uses the full SHS-100K.
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
uv run python scripts\make_manifest.py
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

