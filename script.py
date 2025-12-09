from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from roboflow import Roboflow
except ImportError:  # pragma: no cover - optional dependency
    Roboflow = None


def video_source_type(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run YOLOv11 inference for apple defect detection."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key. Can also be provided via ROBOFLOW_API_KEY env var.",
    )
    parser.add_argument(
        "--workspace",
        default="testroboflow-ttijj",
        help="Roboflow workspace slug containing the project.",
    )
    parser.add_argument(
        "--project",
        default="apple-defect-yozzz-1heaz",
        help="Roboflow project slug to download.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Roboflow dataset version to download.",
    )
    parser.add_argument(
        "--format",
        default="yolov11",
        help="Export format for the Roboflow download.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to an existing data.yaml. If omitted, the dataset will be downloaded.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("yolo11n.pt"),
        help="Path to the pretrained YOLOv11 checkpoint used for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training and inference image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--run-name",
        default="train",
        help="Name of the Ultralytics run directory.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to trained weights to use for inference. Defaults to the latest training run.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run inference.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run real-time webcam inference after training.",
    )
    parser.add_argument(
        "--video-source",
        default=0,
        type=video_source_type,
        help="Video source for inference (int index or path).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum confidence for showing a detection (0–1).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.002,
        help="Minimum bbox area ratio to frame (e.g., 0.002).",
    )
    parser.add_argument(
        "--hue-ratio",
        type=float,
        default=0.10,
        help="Minimum ratio of target hues inside bbox (0–1).",
    )
    parser.add_argument(
        "--defect-conf",
        type=float,
        default=0.35,
        help="Confidence threshold for class=defect.",
    )
    parser.add_argument(
        "--good-conf",
        type=float,
        default=0.50,
        help="Confidence threshold for class=good.",
    )
    parser.add_argument(
        "--disable-heuristics",
        action="store_true",
        help="Disable extra defect heuristics and rely on model only.",
    )
    parser.add_argument(
        "--min-circ",
        type=float,
        default=0.30,
        help="Minimum circularity of hue mask inside bbox (0–1).",
    )
    parser.add_argument(
        "--flesh-ratio",
        type=float,
        default=0.10,
        help="Minimum ratio of bright low-sat non-apple pixels to mark defect.",
    )
    parser.add_argument(
        "--flesh-area",
        type=float,
        default=0.05,
        help="Minimum area ratio of flesh component touching edge to mark bite.",
    )
    parser.add_argument(
        "--sat-min",
        type=int,
        default=70,
        help="Minimum saturation for apple color mask (0–255).",
    )
    parser.add_argument(
        "--exclude-yellow",
        action="store_true",
        help="Exclude yellow hues from apple color mask.",
    )
    return parser.parse_args()


def download_dataset(
    api_key: Optional[str],
    workspace: str,
    project: str,
    version: int,
    export_format: str,
) -> Path:
    if not api_key:
        raise ValueError(
            "Roboflow API key is required to download the dataset. "
            "Provide --api-key or set ROBOFLOW_API_KEY."
        )
    if Roboflow is None:
        raise ImportError(
            "roboflow package is not installed. Install it or supply --data."
        )
    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    dataset = rf_project.version(version).download(export_format)
    return Path(dataset.location) / "data.yaml"


def train_model(
    data_yaml: Path,
    base_model: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    run_name: str,
) -> Path:
    if not base_model.exists():
        raise FileNotFoundError(
            f"Base model checkpoint not found at {base_model}. "
            "Download the desired YOLOv11 weights first or "
            "run with --skip-train to use a pretrained model without fine-tuning."
        )

    # exist_ok=True allows repeated experiments without manual cleanup.
    model = YOLO(str(base_model))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
    )
    weights_path = Path("runs") / "detect" / run_name / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Training finished but {weights_path} was not created. "
            "Check Ultralytics logs for details."
        )
    return weights_path


def load_yolo_model(weights: Path | str) -> YOLO:
    """
    Load a YOLO model.

    If a local path is provided and exists, it is used directly.
    If the file does not exist, the value is passed to Ultralytics, which can
    resolve built-in model names such as 'yolo11n.pt' by auto-downloading them.
    """
    if isinstance(weights, Path):
        if weights.exists():
            return YOLO(str(weights))
        # Fall back to Ultralytics resolver with the string path
        return YOLO(str(weights))
    # weights is a string (e.g. 'yolo11n.pt')
    return YOLO(weights)


def annotate_frame(frame, results, min_area_ratio, hue_ratio_thresh, defect_conf_threshold, good_conf_threshold, disable_heuristics, min_circularity, flesh_ratio_thresh, flesh_area_min_ratio, sat_min, exclude_yellow):
    """
    Draw detections on the frame, highlighting defect apples separately.

    Now that the dataset has two explicit classes:
      - 0 -> Good apple
      - 1 -> Defect apple
    we rely mainly on the model's class prediction, but we also require a higher
    confidence before calling something a defect so that borderline cases are
    treated as good apples instead of false defects.
    """
    good_count = 0
    defect_count = 0

    frame_h, frame_w = frame.shape[:2]
    frame_area = float(frame_h * frame_w)
    min_area = frame_area * float(min_area_ratio)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            raw_label = str(result.names.get(cls, cls))

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = float(w * h)
            if area < min_area:
                continue

            x1c = max(x1, 0)
            y1c = max(y1, 0)
            x2c = min(x2, frame_w)
            y2c = min(y2, frame_h)
            if x2c <= x1c or y2c <= y1c:
                continue

            roi = frame[y1c:y2c, x1c:x2c]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_ch = hsv[:, :, 0]
            s_ch = hsv[:, :, 1]
            sat_mask = s_ch > int(sat_min)
            red1 = np.logical_and(h_ch < 10, sat_mask)
            red2 = np.logical_and(h_ch > 160, sat_mask)
            green = np.logical_and(np.logical_and(h_ch >= 35, h_ch <= 85), sat_mask)
            yellow = np.logical_and(np.logical_and(h_ch >= 20, h_ch <= 35), sat_mask)
            hue_mask = np.logical_or(red1, red2)
            hue_mask = np.logical_or(hue_mask, green)
            if not exclude_yellow:
                hue_mask = np.logical_or(hue_mask, yellow)
            hue_ratio = float(np.mean(hue_mask))
            if hue_ratio < float(hue_ratio_thresh):
                continue

            mask = (hue_mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circ = 0.0
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area_c = float(cv2.contourArea(cnt))
                peri_c = float(cv2.arcLength(cnt, True))
                if peri_c > 0.0:
                    circ = float(4.0 * np.pi * area_c / (peri_c * peri_c))

            v_ch = hsv[:, :, 2]
            low_sat = s_ch < 50
            bright = v_ch > 160
            non_apple = np.logical_not(hue_mask)
            flesh = np.logical_and(np.logical_and(low_sat, bright), non_apple)
            flesh_ratio = float(np.mean(flesh))
            fmask = (flesh.astype(np.uint8)) * 255
            fcontours, _ = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            flesh_area_ratio = 0.0
            touches_edge = False
            if fcontours:
                fcnt = max(fcontours, key=cv2.contourArea)
                farea = float(cv2.contourArea(fcnt))
                roi_area = float(roi.shape[0] * roi.shape[1])
                if roi_area > 0.0:
                    flesh_area_ratio = farea / roi_area
                x_b, y_b, w_b, h_b = cv2.boundingRect(fcnt)
                margin = 3
                touches_edge = (
                    x_b <= margin
                    or y_b <= margin
                    or (x_b + w_b) >= (roi.shape[1] - margin)
                    or (y_b + h_b) >= (roi.shape[0] - margin)
                )

            is_defect_pred = cls == 1 and conf >= defect_conf_threshold
            is_good_pred = cls == 0 and conf >= good_conf_threshold

            heuristic_defect = False
            if (not disable_heuristics) and (not is_defect_pred):
                dark = v_ch < 80
                damage = np.logical_and(dark, low_sat)
                damage_ratio = float(np.mean(damage))
                bite = (
                    flesh_area_ratio > float(flesh_area_min_ratio)
                    and touches_edge
                    and circ < float(min_circularity)
                )
                heuristic_defect = damage_ratio > 0.15 or bite

            is_defect = is_defect_pred or ((not disable_heuristics) and heuristic_defect)
            color = (0, 0, 255) if is_defect else (0, 255, 0)

            if is_defect:
                defect_count += 1
                raw_disp = "Defect apple"
                label = f"DEFECT APPLE ({raw_disp})"
            else:
                good_count += 1
                raw_disp = "Good apple"
                label = f"GOOD APPLE ({raw_disp})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # Show summary counts in the top-left corner
    summary = f"Good: {good_count}  Defect: {defect_count}"
    cv2.putText(
        frame,
        summary,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def run_webcam_inference(model: YOLO, source, imgsz: int, conf: float, min_area_ratio: float, hue_ratio_thresh: float, defect_conf_threshold: float, good_conf_threshold: bool, disable_heuristics: bool, min_circularity: float, flesh_ratio_thresh: float, flesh_area_min_ratio: float, sat_min: int, exclude_yellow: bool):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (imgsz, imgsz))
            # Use user‑configurable confidence so we can control how sure the
            # model must be that it sees an apple at all.
            results = model(resized, conf=conf)
            annotate_frame(resized, results, min_area_ratio, hue_ratio_thresh, defect_conf_threshold, good_conf_threshold, disable_heuristics, min_circularity, flesh_ratio_thresh, flesh_area_min_ratio, sat_min, exclude_yellow)
            cv2.imshow("YOLOv11 Apple Defect Detection", resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def find_local_data_yaml() -> Path | None:
    """
    Try to locate a local data.yaml if Roboflow is not used or fails.
    Checks a few common locations inside this project.
    """
    candidates = [
        Path("data.yaml"),
        Path("Apple-Defect-1") / "data.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    args = parse_args()

    # Resolve data.yaml: prefer explicit CLI path, else Roboflow, else local file.
    data_yaml: Path | None = args.data
    if data_yaml is not None:
        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    else:
        data_yaml = None
        # Try Roboflow first if API key is available and roboflow is installed.
        if args.api_key and Roboflow is not None:
            try:
                data_yaml = download_dataset(
                    api_key=args.api_key,
                    workspace=args.workspace,
                    project=args.project,
                    version=args.version,
                    export_format=args.format,
                )
            except Exception as exc:
                print(
                    f"Warning: Roboflow download failed ({exc}). "
                    "Falling back to local data.yaml search."
                )
        # If Roboflow is not used or failed, fall back to local search.
        if data_yaml is None:
            data_yaml = find_local_data_yaml()
        if data_yaml is None:
            raise FileNotFoundError(
                "Could not find data.yaml. "
                "Pass --data path/to/data.yaml or place it as 'data.yaml' "
                "or 'Apple-Defect-1/data.yaml' in this folder."
            )

    # At this point we have a valid data.yaml Path.
    data_yaml = Path(data_yaml)

    weights: Path | str | None = args.weights
    if weights is None and not args.skip_train:
        # If the base model checkpoint exists locally, fine-tune it.
        if args.base_model.exists():
            weights = train_model(
                data_yaml=data_yaml,
                base_model=args.base_model,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                run_name=args.run_name,
            )
        else:
            # If the base model file is missing, fall back to using the model
            # name directly so Ultralytics can auto-download it instead of
            # raising a hard error.
            print(
                f"Base model checkpoint not found at {args.base_model}. "
                "Skipping training and using the pretrained model name instead."
            )
            weights = str(args.base_model)
    elif weights is None:
        # Use the base model identifier directly; Ultralytics will download
        # it if necessary (e.g. 'yolo11n.pt').
        weights = str(args.base_model)
    model = load_yolo_model(weights)

    if args.webcam:
        run_webcam_inference(
            model,
            args.video_source,
            args.imgsz,
            args.conf,
            args.min_area,
            args.hue_ratio,
            args.defect_conf,
            args.good_conf,
            args.disable_heuristics,
            args.min_circ,
            args.flesh_ratio,
            args.flesh_area,
            args.sat_min,
            args.exclude_yellow,
        )


if __name__ == "__main__":
    main()
