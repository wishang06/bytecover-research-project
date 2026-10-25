# PLAN.md: ByteCover small-data ablation (for the Cursor agent)

Goal: test whether ByteCover's combined loss (Arm B, BNNeck classification + triplet) still beats classification-only (Arm A) as training data shrinks. Both arms share the ResNet50-IBN and GeM encoder, so the comparison is the loss. The paper's "+IBN" row used average pooling; sharing GeM avoids mixing that change into the result. Train on SHS-100K-2025, test on Covers80. Deadline Fri 30 Oct (hard stop Sun 1 Nov).

Rules for the agent
- Andy is learning as he goes. Before each step, explain in plain words what it does and why, then wait for him to say go. After it runs, explain what the output means.
- Do not write the loss functions (classification head, triplet loss, BNNeck), the mAP evaluation or the final analysis. Andy writes those himself. Explain, review and give hints only.
- Use uv for everything (`uv add`, `uv run python ...`). Never pip.
- Run each step, read its output yourself, and fix routine errors yourself.
- Stop and ask Andy only for a real decision, or if a check below fails twice.
- Never commit `audio/`, `cqt/` or `data/shs-100k/` (they're already in .gitignore).
- CQT settings stay fixed: nnAudio CQT2010v2, sr 22050, hop 512, 84 bins, 12 per octave, time averaging `--compress 100` (same as the ByteCover paper).

Done so far
- 300 works sampled with seed 0, 1,498 mp3s in `audio/`, and all 300 works have at least 2 versions.
- Covers80 is in `data/covers80` (160 mp3s).
- Leakage check against full `train.csv` flagged 13 SHS works (`data/exclude_works.txt`). None of them are in the seed-0 sample, so all 300 downloaded works are kept. Training sizes are 50, 150, and 300.

Step 1: cache the training CQTs (running now)
`uv run python scripts\cache_cqt.py --audio audio --out cqt\shs --compress 100`
Good when: it ends with `done: ... 0 failed` and there are about 1,498 .npy files in `cqt\shs`. Rerunning skips files that are already done. If the rate is under about 1 file per 3 s, keep going; if it's much slower, tell Andy and suggest moving to Colab.

Step 2: cache the Covers80 CQTs
`uv run python scripts\cache_cqt.py --audio data\covers80 --out cqt\covers80 --compress 100`
Good when: 160 .npy files and 0 failed.

Step 3: sanity check
Load 3 random .npy files from each folder and print their shape and dtype. Expect (84, about 86) float16, no NaNs, not all zeros.

Step 4: commit the code and metadata (no features)
`git add -A; git commit -m "cqt cached, leakage exclusions"; git push`

Step 5: upload the features for Colab
Zip `cqt\` to `cqt.zip` (about 25 MB) and put it in Andy's Google Drive folder "ByteCover Project" (Drive desktop app or drive.google.com). Report the zip's size.

Step 6: training on Colab (Andy's Google One / Colab)
Open `notebooks/ablation.ipynb` in Colab with a GPU runtime, mount Drive, and unzip `cqt.zip`.
Design:
- Nested training sizes from the 300 kept works: 50, 150 and 300 works (seed-0 order, so smaller sets are subsets of larger ones).
- 2 arms (A: classification only; B: combined loss).
- 5 paired seeds (0 to 4). Each seed is used for both arms at each size, giving 30 runs.
- Metric: Covers80 mAP. Analysis: per-seed difference B minus A at each size, with a 95% t-interval.
- First do one timing run (1 seed, largest size, both arms) and report minutes per run before launching the rest.

Report back to Andy
After each step, give one line: the step, pass or fail, and the key number. Collect the final results table (mean mAP per arm per size, plus the B minus A interval) in `results/summary.csv`.

Locked rules (fixed before seeing any results; do not change)
- Training sizes: 50, 150 and 300 works, nested, seed-0 order. The largest size was corrected from 287 to 300 before any training: the 13 catalog overlaps are not in the seed-0 sample.
- Seeds: 5 paired seeds, 0 to 4, each used for both arms.
- Both arms train for the same fixed number of epochs. No early stopping. The epoch count is still unset: the paper does not state one.
- Batches are 10 works by 3 recordings. Chosen before any results so every size divides evenly and each batch contains positive pairs.
- Covers80 is only for the final test. Never use it for tuning, model selection or picking epochs.
- Pass and fail criteria are set before any results come in, and are not revised afterwards.

