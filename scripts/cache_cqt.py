#!/usr/bin/env python3
"""Cache Orfium-matching CQT features as .npy (float16) for upload to Drive.
Settings copied from orfium/bytecover/models/modules.py (Resnet50.__init__/forward):
  nnAudio CQT2010v2(sr=22050, hop_length=512, n_bins=84, bins_per_octave=12, window='hann',
  output_format='Complex') -> magnitude -> AvgPool over time with compress_ratio=20.
Saved array shape: (84, T//20), the magnitude AFTER compression. NOTE: Orfium applies
time-stretch on the complex CQT BEFORE compression during training, so with cached features
tempo augmentation must be skipped or approximated (e.g. resample the time axis) - decide
before freezing settings. Use --raw to save the uncompressed complex CQT instead (much larger).
  python scripts\\cache_cqt.py --audio audio --out cqt
Needs: pip install torch nnAudio==0.3.2 numpy soundfile  (+ ffmpeg on PATH to decode mp3)."""
import argparse, os, subprocess, sys
import numpy as np

def load_audio(path, sr=22050, seconds=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr)]
    if seconds: cmd += ["-t", str(seconds)]
    cmd += ["-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0: raise RuntimeError(p.stderr.decode(errors="ignore")[-300:])
    return np.frombuffer(p.stdout, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=int, default=200); ap.add_argument("--compress", type=int, default=20)
    ap.add_argument("--raw", action="store_true"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not hasattr(np, 'float'): np.float = float  # nnAudio 0.3.2 uses np.float (removed in numpy>=1.24)
    import torch, nnAudio.features.cqt as nnA
    cqt = nnA.CQT2010v2(sr=22050, hop_length=512, n_bins=84, bins_per_octave=12, window="hann",
                        output_format="Complex", verbose=False).to(a.device)
    pool = torch.nn.AvgPool2d((1, a.compress))
    os.makedirs(a.out, exist_ok=True)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.audio) for f in fs
                   if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm")))
    done = err = 0
    for i, fp in enumerate(files, 1):
        rel = os.path.splitext(os.path.relpath(fp, a.audio))[0].replace(os.sep, "__")
        dst = os.path.join(a.out, rel + ".npy")
        if os.path.exists(dst): continue
        try:
            y = load_audio(fp, seconds=a.seconds)
            with torch.no_grad():
                x = cqt(torch.from_numpy(y.copy()).to(a.device)[None])   # (1, 84, T, 2)
                x = torch.view_as_complex(x.contiguous())
                feat = x[0].cpu().numpy().astype(np.complex64) if a.raw else pool(x.abs())[0].cpu().numpy().astype(np.float16)
            np.save(dst, feat); done += 1
            if i % 50 == 0 or i == len(files): print(f"[{i}/{len(files)}] {rel} {feat.shape}", flush=True)
        except Exception as e:
            err += 1; print("FAIL", fp, e, file=sys.stderr)
    print(f"done: {done} new, {err} failed, {len(files)} audio files total -> {a.out}")

if __name__ == "__main__":
    main()
