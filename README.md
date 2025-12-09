# AI Defect Detector (YOLOv11 Apple Defect Detection)

Real‑time detector that classifies apples as `Good` or `Defect` using a YOLOv11 model plus targeted color/shape heuristics. Supports webcam inference and fine‑tuning on your dataset.

## Quick Start (Windows)
- Prerequisites:
  - Install Python 3.10+ and pip
  - Optional: Git for version control
- Create and activate a virtual environment:
  ```powershell
  cd "c:\Users\misterr\Downloads\Apple Defect.v1i.yolov11"
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
- Install dependencies (CPU):
  ```powershell
  pip install --upgrade pip
  pip install torch torchvision torchaudio
  pip install ultralytics opencv-python roboflow
  ```
- Run webcam inference with existing weights:
  ```powershell
  .\.venv\Scripts\python.exe script.py --skip-train --weights runs\detect\apples-2class\weights\best.pt --webcam --video-source 0 --imgsz 640 --conf 0.30 --min-area 0.0015 --hue-ratio 0.12 --min-circ 0.45 --flesh-ratio 0.08 --flesh-area 0.05 --defect-conf 0.40 --good-conf 0.55 --sat-min 90 --exclude-yellow
  ```
  If you don’t have trained weights yet, you can start with the base model:
  ```powershell
  .\.venv\Scripts\python.exe script.py --skip-train --weights yolo11n.pt --webcam --video-source 0 --imgsz 640 --conf 0.35 --disable-heuristics
  ```

## Quick Start (macOS)
- Prerequisites:
  - Install Command Line Tools: `xcode-select --install`
  - Optional: Homebrew for easier installs: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- Install Python 3 and create venv:
  ```bash
  # If using Homebrew
  brew install python@3.11
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies (CPU):
  ```bash
  pip3 install --upgrade pip
  pip3 install torch torchvision torchaudio
  pip3 install ultralytics opencv-python roboflow
  ```
- Allow camera permissions:
  - System Settings → Privacy & Security → Camera → enable for Terminal/iTerm.
- Run webcam inference (replace weights path if different):
  ```bash
  python3 script.py --skip-train --weights runs/detect/apples-2class/weights/best.pt --webcam --video-source 0 --imgsz 640 --conf 0.30 --min-area 0.0015 --hue-ratio 0.12 --min-circ 0.45 --flesh-ratio 0.08 --flesh-area 0.05 --defect-conf 0.40 --good-conf 0.55 --sat-min 90 --exclude-yellow
  ```

## Training
- Prepare/merge the dataset for two classes (`Good apple`, `Defect apple`):
  ```bash
  python script.py --help
  python prepare_combined_dataset.py
  ```
- Train a 2‑class model (CPU example):
  ```bash
  python script.py --data data.yaml --base-model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --run-name apples-2class
  ```
- After training, test with webcam:
  ```bash
  python script.py --skip-train --weights runs/detect/apples-2class/weights/best.pt --webcam --video-source 0 --imgsz 640 --conf 0.30
  ```

## CLI Flags (key ones)
- `--conf`: Minimum detection confidence (default 0.35)
- `--min-area`: Minimum bbox area ratio to frame
- `--hue-ratio`: Minimum ratio of apple‑like hues in bbox
- `--min-circ`: Minimum circularity (roundness) of hue mask
- `--flesh-ratio`: Bright, low‑sat non‑apple ratio (bite indicator)
- `--flesh-area`: Minimum connected “bite” area that touches the edge
- `--defect-conf`: Confidence threshold for class `Defect`
- `--good-conf`: Confidence threshold for class `Good`
- `--sat-min`: Minimum saturation for color mask; raise to filter skin/orange
- `--exclude-yellow`: Exclude yellow hues from apple color mask
- `--disable-heuristics`: Rely solely on model predictions (no color/shape gates)

## Troubleshooting
- CSV permission error on Windows during training:
  - Close any app viewing `runs\detect\<run>\results.csv` or change `--run-name`.
- Webcam not opening on macOS:
  - Grant camera permission to Terminal/iTerm and restart.
- Faces/oranges detected as apples:
  - Increase `--sat-min` (100–120), `--hue-ratio` (0.15–0.20), `--min-circ` (0.60–0.65), or use `--exclude-yellow`.
- Bites not detected:
  - Raise `--flesh-area` (0.08–0.12) and lower `--defect-conf` (0.35–0.40).

## Project Structure
- `script.py`: Training and webcam inference with configurable filters
- `data.yaml`: 2‑class dataset config (Good apple, Defect apple)
- `prepare_combined_dataset.py`: Merges and relabels dataset into combined layout
- `runs/`: Ultralytics training outputs (ignored by `.gitignore`)
- `Apple-Defect-1/`, `train/`, `combined/`: dataset folders (large files ignored)

## GitHub
- `.gitignore` excludes datasets, runs, venv, and weights to keep the repo small.
- To publish:
  ```bash
  git init
  git add .
  git commit -m "Initial commit: apple defect detector"
  git branch -M main
  git remote add origin https://github.com/<your-user>/<repo>.git
  git push -u origin main
  ```
