# Windows setup (Andy's laptop)

Every command below goes in **PowerShell**. You don't need bash or WSL.

## 1. Install Python, ffmpeg and Git (once)
```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Gyan.FFmpeg
winget install -e --id Git.Git
```
**Close PowerShell and open a new window** so the PATH change takes effect. Then check:
```powershell
python --version      # 3.11.x
ffmpeg -version       # first line prints the version
```
If `python` opens the Microsoft Store instead, go to Settings > Apps > Advanced app settings > App execution aliases, turn off the python aliases, and use `py` in place of `python`.

## 2. Get the project and the Python packages
Copy the "ByteCover Project" folder from Google Drive to e.g. `C:\bytecover-project`. Then:
```powershell
cd C:\bytecover-project
git clone --depth 1 https://github.com/second-hand-songs/shs-100k data\shs-100k
python -m pip install --upgrade pip
python -m pip install -U "yt-dlp[default]"
```
yt-dlp gets updated often, and YouTube breaks old versions. Rerun `python -m pip install -U "yt-dlp[default]"` if downloads start failing with extractor errors.

## 3. Tonight: download audio (resumable, takes a few hours)
```powershell
python scripts\download_audio.py --csv data\shs-100k\train.csv --out audio
```
- If you stop it (Ctrl+C), running it again resumes. Finished files are skipped, and failed IDs are logged in `audio\failures.csv` and not tried again (add `--retry-failed` to retry them).
- It stops by itself if YouTube shows a bot check 5 times in a row. Wait a few hours, then rerun the same command.
- At the end it prints `works with >=2 versions: N / 300`. You need N >= 200.
- **Optional, your choice:** `--cookies-from-browser firefox` (or `chrome`) makes yt-dlp use your browser's YouTube login. It's off by default. Only use it if you're comfortable with that; a throwaway Google account is safer.
- Keep the laptop plugged in and stop it from sleeping (Settings > System > Power).

## 4. Covers80 + leakage check
```powershell
python scripts\get_covers80.py --out data\covers80
python scripts\leakage_check.py --lists data\covers80\list1.txt data\covers80\list2.txt
```
Open `data\leakage_matches.csv` and delete the false matches (for example "Yesterday" vs "Yesterdays") from `data\exclude_works.txt`.

## 5. CQT cache (after downloading)
```powershell
python -m pip install torch numpy nnAudio==0.3.2
python scripts\cache_cqt.py --audio audio --out cqt\shs
python scripts\cache_cqt.py --audio data\covers80 --out cqt\covers80
```
Upload `cqt\`, `audio\downloaded.csv` and `data\exclude_works.txt` to the Drive folder.
