# CQT math reference

A cached `.npy` file is one recording as a constant-Q spectrogram. This note is the arithmetic behind its shape. The study design (training sizes, seeds, arms) is in `README.md` and `PLAN.md`.

Paper source for the feature itself: Du et al., *ByteCover*, ICASSP 2021, [arXiv:2010.14022](https://arxiv.org/abs/2010.14022), section 2.1.

## Shape

```
x.shape == (84, T)
x.dtype == float16
```

| Axis | Size | Meaning |
|---|---|---|
| Rows | 84 | Pitch. One semitone per row. |
| Columns | T | Time, after averaging. |
| Cell | `float16`, at least 0 | Magnitude: how strong that pitch is in that time slice. |

`T` is not stored anywhere else. It is `x.shape[1]`, and it changes with the length of the clip.

The paper writes the same object as \(S \in \mathbb{R}^{84 \times T}\).

## Frequency axis (the 84)

The paper sets 84 bins, 12 bins per octave, and a Hann window. It does not state the lowest frequency.

`scripts/cache_cqt.py` does not pass `fmin`, so nnAudio `CQT2010v2` uses its default. Bin \(k\) is

```
f(k) = 32.70 * 2^(k / 12) Hz
k = 0, 1, ..., 83
```

32.70 Hz is the note C1. nnAudio's docstring calls 32.70 Hz "C0". That label is wrong: C0 is 16.35 Hz.

| k | Note | Hz |
|---|---|---|
| 0 | C1 | 32.70 |
| 12 | C2 | 65.40 |
| 24 | C3 | 130.80 |
| 36 | C4 (middle C) | 261.60 |
| 48 | C5 | 523.20 |
| 60 | C6 | 1046.40 |
| 72 | C7 | 2092.80 |
| 83 | B7 | 3950.7 |

The sample rate is 22050 Hz, so the Nyquist frequency is \(22050 / 2 = 11025\) Hz. The top bin sits below that.

`imshow(..., origin="lower")` puts bin 0 at the bottom of the picture. A held note shows up as a bright horizontal ridge.

Low rows use a longer analysis window than high rows. That is the constant-Q tradeoff: every row is one semitone, and the bottom of the picture is blurrier in time.

```
Q = 1 / (2^(1/12) - 1) ≈ 16.82
window at bin k ≈ Q * 22050 / f(k) samples
```

At C1 that window is about 0.51 s. At B7 it is about 4 ms.

## Time axis (T)

```
sample rate           sr = 22050 Hz
hop                   512 samples
raw frame spacing     512 / 22050 = 0.02322 s ≈ 23.2 ms
averaging factor      100
saved column spacing  100 * 512 / 22050 = 2.322 s
```

The hop is the gap between raw CQT frames. The cache then replaces every 100 raw frames with their average. Those 100-frame blocks do not overlap, and a leftover tail shorter than 100 frames is dropped.

```
T ≈ floor( (duration_seconds * 22050 / 512) / 100 )
duration_seconds ≈ T * 100 * 512 / 22050 = T * 2.322
```

Both the downloader and the cache cut audio at 200 seconds:

```
200 * 22050 = 4,410,000 samples
4,410,000 / 512 ≈ 8,613 raw frames
8,613 / 100 = 86.13  →  T ≈ 86
shape ≈ (84, 86)
```

nnAudio pads the waveform before the transform, so a real full-length file can be a frame or two away from 86. The sanity check prints the widths that are actually on disk.

A shorter video produces a smaller `T`. Sixty seconds is about 2,584 raw frames, so \(T \approx 25\) and the shape is about `(84, 25)`. Reading a file the other way: shape `(84, 40)` is about \(40 \times 2.322 \approx 93\) seconds of audio.

## Storage

`float16` is this project's storage choice. The paper does not specify a dtype. Two bytes per cell keeps the cache small enough to upload:

```
one full clip ≈ 84 * 86 * 2 bytes ≈ 14.4 KB
about 1,500 full clips ≈ 20 MB
```

That is why the zip of `cqt/` is expected to be around 25 MB. Shorter clips take less.

## Where each number is fixed

| Number | Where it comes from |
|---|---|
| 22050 Hz, hop 512, Hann window, 12 bins/octave, 84 bins, average by 100, shape \(84 \times T\) | ByteCover paper, section 2.1 |
| Lowest frequency 32.70 Hz | nnAudio default. The cache script does not set `fmin`. |
| 200 s maximum, mono | `scripts/download_audio.py` and `cache_cqt.py --seconds` (default 200) |
| `--compress` default of 20 inside `cache_cqt.py` | Orfium's code default. The files on disk were cached with `--compress 100`. A rerun that omits that flag comes out about 430 columns wide. |
| `float16` | `np.save` in `scripts/cache_cqt.py` |

## The picture

An SHS spectrogram is a plot of one array in `cqt/shs/` (a Second Hand Songs training clip). Rows are pitch, columns are time, color is magnitude. The same plot can be made for a file in `cqt/covers80/`. The image is an eye check that the array is music. Training reads the `.npy` files.
