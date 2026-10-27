# Report outline

A plan for the write-up Andy will take to office hours with Professor Edgar Dobriban (STAT 4310). This is not the report. It is the structure, the points to make, and the questions to ask. Every number below is copied from `results/summary.csv`, `results/runs.csv`, `results/train_loss.csv`, or the training code. Nothing here is a new experimental result.

The numeric pass/fail thresholds are not in the repo. `PLAN.md` says they were fixed before any Covers80 score and must not be revised afterwards. It does not record the actual cutoffs. The report must not invent them after seeing the table.

## Title

- Working title: Does ByteCover's triplet loss still help when the training set is small?
- Subtitle idea: A 30-run ablation on nested subsets of SHS-100K-2025, tested on Covers80.
- Author line: Wang Ieong Shang. The model is from Du et al., ICASSP 2021. The small-data comparison is this project.

## Abstract

- One paragraph. State the question, the two arms, the three nested sizes, the test set, and the paired result.
- Question: if both arms share the same encoder, does adding a batch-hard triplet loss (margin 0.3, on f_t) raise Covers80 mAP when training data is scarce?
- Design in one sentence: 3 sizes (50, 150, 300 works) x 2 arms x 5 seeds = 30 runs, 100 epochs, no tuning on Covers80.
- Result in one sentence, using `results/summary.csv` (means over 5 paired seeds, 95% paired t-interval, 4 degrees of freedom, critical value 2.776445):
  - 50 works: mean mAP A 0.032976, B 0.048001, mean B minus A 0.015025 (sd 0.006630), interval 0.006793 to 0.023257.
  - 150 works: mean mAP A 0.045077, B 0.055848, mean B minus A 0.010772 (sd 0.018329), interval -0.011987 to 0.033530.
  - 300 works: mean mAP A 0.056204, B 0.079402, mean B minus A 0.023198 (sd 0.017189), interval 0.001855 to 0.044541.
- The intervals at 50 and 300 works lie entirely above zero. The interval at 150 works crosses zero.
- Close by saying absolute mAP stays near the floor the README states for a random ranking of the other 159 tracks (about 0.035). The claim is about the paired difference, and that difference is small.

## Introduction and motivation

- Cover song identification: two recordings can be the same work and still differ in key, tempo, instruments, or structure.
- ByteCover trains a ResNet-IBN with a classification loss and a triplet loss, joined by a BNNeck. The paper reports that the combination beats classification alone on the full SHS-100K.
- This study asks whether that extra loss still helps on a few hundred works, which is the setting a small replication can actually train.
- Say what was locked before any Covers80 score: 100 epochs, no early stopping, Covers80 used only as the final test, same Colab NVIDIA L4 for every counted run.
- Say what this section will not do: it will not treat the README rounding as the source. The README prints 0.0330, 0.0480, 0.0150 at 50 works, and the matching rounded rows at 150 and 300. The report should quote `results/summary.csv`.

## Related work

- Du, Yu, Zhu, Chen, and Ma, *ByteCover: Cover Song Identification via Multi-Loss Training*, ICASSP 2021, arXiv:2010.14022. This is the method being replicated. Cite what the paper trains (classification plus triplet, BNNeck) and that their data scale is the full SHS-100K, not 300 works.
- SHS-100K (Xu et al., 2018) as the source of works, and this project's use of the SHS-100K-2025 list. Some original YouTube links are gone, which is why the download is 1,498 recordings rather than a complete catalog.
- Covers80 (Ellis, 2007): 80 works, two versions each. The test uses the 160 recordings named in `data/covers80/list1.txt` and `list2.txt`.
- Orfium's open-source ByteCover reimplementation, used as a reference for the CQT settings and for the choice of 100 epochs (the paper itself states no epoch count).
- One short paragraph on what this project changes on purpose: nested training sizes, a shared GeM encoder in both arms, and a batch of 10 works by 3 clips so that 50, 150, and 300 all divide evenly. The paper's batch of 32 does not.

## Method

### Data

- Training pool: SHS-100K-2025, 300 works sampled with seed 0, 1,498 recordings in `data/train_manifest.csv`. Every work has between 3 and 5 recordings. The batch sampler needs at least 3.
- Nested sizes: 50, then 150, then 300 works. Each size is a prefix of `data/work_order.txt`, so the smaller set sits inside the larger one. Training seeds do not reshuffle that list.
- Leakage: `data/exclude_works.txt` lists 13 SHS works that fuzzy-matched Covers80. None of those 13 ids are in `data/work_order.txt`, so the training pool keeps all 300 works.
- Audio: `yt-dlp`, first 200 seconds, mono, 22,050 Hz.

### CQT

- nnAudio CQT2010v2, 84 bins, 12 per octave, hop 512, time averaging of 100 frames (`--compress 100`), saved as float16.
- Each clip is zero-padded to 86 frames. The true length is stored and GeM skips the padded time columns.
- Cached Covers80 features: the notebook checks for 164 `.npy` files. Scoring uses only the 160 list files. See Limitations for the PLAN.md mismatch.

### Model

- Shared by both arms: ResNet50-IBN. The first three stages use IBN. Stage 4 stays plain batch norm and is not downsampled (stride 1), so a padded 84 by 86 CQT becomes a feature map of 2048 by 6 by 6.
- GeM pooling to f_t, dimension 2048.
- BNNeck: BatchNorm1d maps f_t to f_c. A bias-free linear layer classifies f_c. The number of outputs equals the training size for that run (50, 150, or 300).
- Retrieval uses f_c for both arms. The evaluation code calls `head.embed`, which is the BatchNorm, not the classifier.

### Arm A and Arm B

- Arm A: softmax cross-entropy on f_c only.
- Arm B: the same cross-entropy plus a batch-hard triplet on f_t (Euclidean distance, margin 0.3). The hard positive is the farthest other clip of the same work. The hard negative is the nearest clip of any other work.
- The encoder, GeM, BNNeck, and classifier are the same in both arms. The loss is the comparison. An earlier Arm A without the BNNeck was discarded. Those scores are in `results/runs_discarded_biased_head.csv` and should be mentioned only as a discarded run, not mixed into the table.

### Batching

- P x K batches: 10 works by 3 clips = 30 clips. Every work in that run's size is in exactly one batch per epoch, so each triplet has a positive pair.
- Batches per epoch: 5 at 50 songs, 15 at 150 songs, 30 at 300 songs.
- Over 100 epochs that is 500, 1,500, and 3,000 Adam updates. The 300-work runs take 6 times as many steps as the 50-work runs. A fixed epoch count is not a fixed step count. This is one of the questions for Professor Dobriban.

### Training setup

- Adam, learning rate 0.0004, weight decay 0. No learning-rate schedule.
- 100 epochs, no early stopping. `PLAN.md` locks 100 from a timing run of 0.18 minutes per epoch at size 300, seed 0, before any Covers80 score.
- Each run builds a new encoder with `torch.manual_seed(seed)` and shuffles batches with that same seed. Seeds are 0, 1, 2, 3, and 4, paired across arms.
- A run counts only on a Colab NVIDIA L4. All 30 rows in `results/runs.csv` and all 3,000 rows in `results/train_loss.csv` record `NVIDIA L4`.
- Checkpoints are named like `A_s300_k4.pt` and were written to Google Drive (`ByteCover Project/results`). The `.pt` files are gitignored. `results/runs.csv`, `results/train_loss.csv`, and `results/summary.csv` are in the repo.

## Evaluation

- Covers80 is the final test only. It is not used to pick epochs, architectures, or hyperparameters.
- 80 works, 2 versions, 160 recordings. Line i of `list1.txt` is paired with line i of `list2.txt`.
- Each recording is a query against the other 159, ranked by cosine similarity of f_c. There is one relevant track, so average precision is 1 over that track's rank. mAP is the mean over the 160 queries.
- The metric in `results/runs.csv` is this mAP. There is no top-1 column and no other test set.
- There is no validation split. Epochs were not chosen by watching a held-out set.

## Results

Use this table from `results/summary.csv`. n = 5 paired seeds at every size. df = 4. t critical value = 2.776445.

| Works | Mean mAP A | Mean mAP B | Mean B minus A | sd of the five deltas | 95% interval |
|---|---|---|---|---|---|
| 50 | 0.032976 | 0.048001 | 0.015025 | 0.006630 | 0.006793 to 0.023257 |
| 150 | 0.045077 | 0.055848 | 0.010772 | 0.018329 | -0.011987 to 0.033530 |
| 300 | 0.056204 | 0.079402 | 0.023198 | 0.017189 | 0.001855 to 0.044541 |

- Per-seed B minus A, from `results/runs.csv`, so the interval widths have a source:
  - 50 works: 0.021525, 0.007267, 0.021286, 0.009233, 0.015813. All five are positive.
  - 150 works: 0.032118, 0.022237, -0.012280, -0.003535, 0.015318. Seeds 2 and 3 go the other way, which is why the interval crosses zero and the sd is 0.018329.
  - 300 works: 0.028320, 0.021310, 0.043391, -0.003843, 0.026813. Seed 3 is slightly negative. The other four are positive, and the interval stays above zero.
- Mean mAP rises with the number of works for both arms, and stays close to the README's random-ranking figure of about 0.035. At 50 works, Arm A's mean (0.032976) is just under that figure and Arm B's mean (0.048001) is just over it.
- Do not convert "interval excludes zero" into a pass or a fail. The thresholds that were supposed to make that decision are not written down.

## Loss convergence

`results/train_loss.csv` has one row per run per epoch: arm, size, seed, epoch, mean_objective, gpu. That is 30 runs x 100 epochs = 3,000 rows, all on NVIDIA L4. The repo logs the curves so a run can be shown to have levelled off without looking at Covers80. It does not contain that written check. The averages below are the five seeds' logged `mean_objective`, computed for this outline.

| Arm | Works | Epoch 1 | Epoch 50 | Epoch 100 |
|---|---|---|---|---|
| A | 50 | 4.128705 | 0.002778 | 0.000587 |
| A | 150 | 5.240091 | 0.002414 | 0.000776 |
| A | 300 | 5.945428 | 0.001887 | 0.000837 |
| B | 50 | 8.869774 | 0.159725 | 0.141571 |
| B | 150 | 8.435413 | 0.183865 | 0.105981 |
| B | 300 | 8.520470 | 0.210222 | 0.136060 |

- Arm A's objective is cross-entropy only. By epoch 50 the five-seed mean is already below 0.003 at every size, and by epoch 100 it is below 0.001.
- Arm B's objective is cross-entropy plus the triplet hinge. At epoch 1 the five-seed means are 8.869774, 8.435413, and 8.520470. At epoch 100 they are 0.141571, 0.105981, and 0.136060. From epoch 50 to epoch 100 the curve still moves, especially at 150 works (0.183865 down to 0.105981).
- The two columns are different functions. A smaller Arm A number does not mean Arm A trained "more". The report should plot each arm against its own early epochs, and should not rank the arms by the raw objective.
- Ask Professor Dobriban whether these curves are flat enough to treat epoch 100 as past the training transient, given that Arm B is still drifting between epoch 50 and epoch 100.

## Discussion

- Lead with the paired differences, not the absolute mAP. The triplet term is associated with a higher mean mAP at every size. The uncertainty is large: at 150 works the interval includes differences below zero, and at 300 works the lower end is 0.001855.
- The pattern is not a clean "more data, more help". The mean delta is 0.015025 at 50, 0.010772 at 150, and 0.023198 at 300. With five seeds, that dip can be noise. Two of the five 150-work pairs are negative.
- Put the step-count confound next to that sentence. The 50-work runs saw 500 updates and the 300-work runs saw 3,000. A larger gain at 300 works is also a gain after six times as many steps.
- Absolute performance: even the best cell in the summary, mean mAP B at 300 works (0.079402), is a weak ranker on a 160-track set whose chance level the README puts near 0.035. Say that plainly so the paired result is not read as "the system works".
- The shared encoder means the comparison is the loss. Mention the paper's "+IBN" row used average pooling, and this study shares GeM so a pooling change is not mixed into B minus A.

## Limitations

- No validation set. Covers80 is the only held-out music, and it is the final test. There is no split for choosing epochs or the margin.
- Only mAP is reported. No top-1, no mean rank, no second dataset.
- Five seeds. The t-interval uses 4 degrees of freedom and the critical value 2.776445, so the intervals are wide. The 150-work sd of the deltas is 0.018329, larger than the mean delta 0.010772.
- Fixed epochs, unequal steps: 500, 1,500, and 3,000 updates.
- Scale. 300 works and 1,498 recordings, against the paper's full SHS-100K.
- No tempo augmentation. CQTs are cached, so the paper's augmentation on the raw complex CQT is not available.
- Pass/fail cutoffs are missing. `PLAN.md` says they were pre-registered. They are not in PLAN, the README, or the code. The report cannot claim a pre-registered pass or fail until those numbers are recovered from notes taken before the results. If they cannot be recovered, say that the decision rule was not preserved.
- Code and doc mismatches to disclose, not to hide:
  - `bytecover/evaluate.py` docstring says the retrieval vector is f_t for Arm A and f_c for Arm B. `embed_paths` uses `head.embed` for both arms, which is f_c. `bytecover/loss.py` and the README agree with the code path.
  - `PLAN.md` step 2 says the Covers80 cache is good at 160 `.npy` files. The notebook requires 164 files. `evaluate.py` says the four extra files are not scored. The reported mAP uses the 160 list recordings.
- An earlier Arm A without the BNNeck was thrown out. Do not pool it with the corrected Arm A.

## Future work

- Only list things this repo has not done.
- More than five seeds, or a different test of the paired differences, if Professor Dobriban thinks the t-interval is too thin.
- A second test set. Covers80 is 160 recordings. Da-TACOS is a candidate to ask about. This repo has no Da-TACOS numbers.
- A step-matched rerun: the same number of Adam updates at 50, 150, and 300 works, so the small runs are not stopped after 500 steps.
- Write down a decision rule before any new test scores, and keep the rule in the repo.
- Tempo augmentation, or training on more of SHS-100K, only if the question is still the loss and not a new model.

## Reproducibility

- Code: `bytecover/model.py`, `bytecover/loss.py`, `bytecover/train.py`, `bytecover/evaluate.py`, `bytecover/grid.py`. Summary script: `scripts/summarize.py`, which rewrites `results/summary.csv` from `results/runs.csv`.
- Data order: `data/work_order.txt`, `data/train_manifest.csv`, `data/covers80/list1.txt`, `data/covers80/list2.txt`, `data/exclude_works.txt`.
- Locked constants: sizes 50, 150, 300; seeds 0 to 4; 100 epochs; batch 10 by 3; Adam learning rate 0.0004; weight decay 0; triplet margin 0.3; CQT as in the README setup block.
- Hardware rule: Colab NVIDIA L4. The gpu column is the check.
- What a reader does not get from git: the audio, the CQT cache, and the `.pt` checkpoints (Google Drive folder `ByteCover Project`). The csv results are in `results/`.
- Point at `docs/diagrams.md` for the flowchart and the training-structure figure.

## Questions for Professor Edgar Dobriban

Office hours, STAT 4310. Eight questions. The first four are the ones to ask even if time is short.

1. Is a paired t-interval with 5 seeds per size enough? The intervals use 4 degrees of freedom and a critical value of 2.776445. At 150 works the sd of B minus A is 0.018329 and the interval is -0.011987 to 0.033530. Would you want more seeds, a different interval, or a different test of the five paired differences?
2. Is Covers80 too small at 160 recordings (each query ranked against the other 159)? Should a second test set be added, for example Da-TACOS, before any claim about cover-song identification rather than about this one list?
3. Does a fixed epoch count treat the small runs unfairly? They get 500 Adam steps, the 150-work runs get 1,500, and the 300-work runs get 3,000. Should a follow-up fix the number of steps instead of the number of epochs?
4. What would make this worth extending into real research with you, rather than leaving it as a small reproduction of ByteCover? Which piece is missing: a pre-registered decision rule, a second dataset, a step-matched comparison, or a different question?
5. At 150 works, two of the five paired deltas are negative (-0.012280 and -0.003535), while all five at 50 works are positive. Should that dip be described as noise from n = 5, or is it a result that needs its own sentence?
6. Arm A's cross-entropy is below 0.001 by epoch 100 (0.000587, 0.000776, 0.000837). Arm B's combined objective at epoch 100 is 0.141571, 0.105981, and 0.136060, and at 150 works it moves from 0.183865 at epoch 50 to 0.105981 at epoch 100. The two logs are different functions. How should the report talk about convergence, and is epoch 100 late enough for Arm B?
7. Absolute mAP sits near the README's random-ranking floor of about 0.035 (best summary cell: 0.079402). Should the abstract lead with the paired intervals and explicitly refuse a claim that the system identifies covers well?
8. `PLAN.md` says the pass and fail lines were fixed before the results and must not be changed. The numbers are not in the repo. If they cannot be found in notes from before the runs, should the report say the decision rule was not preserved, rather than writing a cutoff now?
