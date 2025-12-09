from __future__ import annotations

"""
Merge defect-apple images from:
  ../apple defect detection.v1i.yolov11
into this project’s existing dataset:
  ./train/images, ./train/labels

Result:
  - class 0: Good apple  (your original labels stay as‑is)
  - class 1: Defect apple (all objects from the defect dataset are rewritten to class 1)

Run once from inside this folder:
    python merge_defect_into_train.py
Then retrain:
    python script.py --data data.yaml --epochs 50
"""

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT / "train"
DEFECT_ROOT = ROOT.parent / "apple defect detection.v1i.yolov11"


def rewrite_labels_to_class1(src_lbl_dir: Path, dst_lbl_dir: Path):
    """Read YOLO labels and change every class id to 1 (defect)."""
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for lbl_path in src_lbl_dir.glob("*.txt"):
        text = lbl_path.read_text().strip()
        if not text:
            continue
        new_lines = []
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            parts[0] = "1"  # force class id to 1
            new_lines.append(" ".join(parts))
        (dst_lbl_dir / lbl_path.name).write_text("\n".join(new_lines) + "\n")


def copy_defect_split_into_train(split: str):
    """
    Copy images/labels from each split (train/valid/test) of the defect dataset
    into this project’s ./train/images and ./train/labels.
    """
    src_split = DEFECT_ROOT / split
    if not src_split.exists():
        return

    src_img = src_split / "images"
    src_lbl = src_split / "labels"

    dst_img = TRAIN_ROOT / "images"
    dst_lbl = TRAIN_ROOT / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for img_path in src_img.glob("*.jpg"):
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        shutil.copy2(img_path, dst_img / img_path.name)

    # Rewrite labels to class 1 into the train labels folder
    rewrite_labels_to_class1(src_lbl, dst_lbl)


def update_data_yaml():
    """
    Update root data.yaml so it still points to ./train/images
    but declares two classes: Good apple and Defect apple.
    """
    data_yaml = ROOT / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found at {data_yaml}")

    content = """train: ../train/images
val: ../train/images  # using train as val for now

nc: 2
names: ['Good apple', 'Defect apple']
"""
    data_yaml.write_text(content)


def main():
    if not TRAIN_ROOT.exists():
        raise SystemExit(f"Expected existing dataset at {TRAIN_ROOT} but it was not found.")
    if not DEFECT_ROOT.exists():
        raise SystemExit(
            f"Expected defect dataset folder at {DEFECT_ROOT} but it was not found.\n"
            "Make sure 'apple defect detection.v1i.yolov11' is next to this folder."
        )

    print("Merging defect apples into existing train/ folder...")
    for split in ("train", "valid", "test"):
        copy_defect_split_into_train(split)

    print("Updating data.yaml for 2 classes (Good apple, Defect apple)...")
    update_data_yaml()

    print("Done.")
    print("Now retrain with:")
    print("  python script.py --data data.yaml --epochs 50")


if __name__ == "__main__":
    main()


