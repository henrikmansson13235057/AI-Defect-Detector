from __future__ import annotations

"""
Utility script to combine:
- good apple dataset in this folder (`train/images`, `train/labels`)
- defect apple dataset in `../apple defect detection.v1i.yolov11`

into a single 2‑class YOLO dataset:
  class 0 -> Good apple
  class 1 -> Defect apple

Run once:
    python prepare_combined_dataset.py
Then retrain:
    python script.py --data data.yaml --epochs 50
"""

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOOD_ROOT = ROOT / "Apple-Defect-1" / "train"
DEFECT_ROOT = ROOT.parent / "apple defect detection.v1i.yolov11"
COMBINED_ROOT = ROOT / "combined"


def copy_good():
    """Copy existing 'Good apple' images/labels as class 0."""
    src_img = GOOD_ROOT / "images"
    src_lbl = GOOD_ROOT / "labels"
    dst_img = COMBINED_ROOT / "train" / "images"
    dst_lbl = COMBINED_ROOT / "train" / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for img_path in src_img.glob("*.jpg"):
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            # Skip images without labels
            continue
        shutil.copy2(img_path, dst_img / img_path.name)
        shutil.copy2(lbl_path, dst_lbl / lbl_path.name)


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
            # Force class id to 1; keep bbox values as‑is
            parts[0] = "1"
            new_lines.append(" ".join(parts))
        (dst_lbl_dir / lbl_path.name).write_text("\n".join(new_lines) + "\n")


def copy_defect_split(split: str):
    """
    Copy defect images/labels from the Roboflow dataset.
    We treat ALL objects as class 1 (Defect apple).
    """
    src_split = DEFECT_ROOT / split
    if not src_split.exists():
        return

    src_img = src_split / "images"
    src_lbl = src_split / "labels"

    dst_img = COMBINED_ROOT / "train" / "images"
    dst_lbl = COMBINED_ROOT / "train" / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for img_path in src_img.glob("*.jpg"):
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        shutil.copy2(img_path, dst_img / img_path.name)

    # Rewrite labels to class 1 into the combined labels folder
    rewrite_labels_to_class1(src_lbl, dst_lbl)


def write_combined_data_yaml():
    """
    Overwrite the root data.yaml to describe the new 2‑class dataset.
    """
    data_yaml = ROOT / "data.yaml"
    content = """train: ./combined/train/images
val: ./combined/train/images  # using train as val for now

nc: 2
names: ['Good apple', 'Defect apple']
"""
    data_yaml.write_text(content)


def main():
    if not GOOD_ROOT.exists():
        raise SystemExit(f"Expected good apple dataset at {GOOD_ROOT} but it was not found.")
    if not DEFECT_ROOT.exists():
        raise SystemExit(
            f"Expected defect dataset folder at {DEFECT_ROOT} but it was not found.\n"
            "Make sure 'apple defect detection.v1i.yolov11' is next to this folder."
        )

    print("Creating combined dataset folder...")
    (COMBINED_ROOT / "train" / "images").mkdir(parents=True, exist_ok=True)
    (COMBINED_ROOT / "train" / "labels").mkdir(parents=True, exist_ok=True)

    print("Copying good apples (class 0)...")
    copy_good()

    print("Copying defect apples (class 1) from train/valid/test splits...")
    for split in ("train", "valid", "test"):
        copy_defect_split(split)

    print("Writing updated data.yaml for 2‑class problem...")
    write_combined_data_yaml()

    print("Done.")
    print("Now retrain with:")
    print("  python script.py --data data.yaml --epochs 50")


if __name__ == "__main__":
    main()


