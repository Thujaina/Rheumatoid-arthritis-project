"""YOLOv8 joint ROI training and inference utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def train_joint_detector(data_yaml: str | Path, model_name: str = "yolov8n.pt",
                        epochs: int = 100, image_size: int = 640,
                        batch_size: int = 16, device: str | int | None = None,
                        project: str = "runs/joint_roi", name: str = "yolov8") -> Any:
    """Fine-tune an Ultralytics detector. YAML labels should include wrist/MCP/PIP."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics: pip install ultralytics") from exc
    yaml_path = Path(data_yaml)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {yaml_path}")
    detector = YOLO(model_name)
    return detector.train(data=str(yaml_path), epochs=epochs, imgsz=image_size,
                          batch=batch_size, device=device, project=project, name=name)


def crop_joint_rois(image_path: str | Path, model_path: str | Path = "best.pt",
                    confidence: float = 0.25, output_dir: str | Path | None = None,
                    image_size: int = 640) -> list[dict[str, Any]]:
    """Detect joint boxes and return ROI records (class, confidence, box, image, path).

    When no detector box is found, returns an empty list; callers can report that
    manual annotation/detector training is required instead of silently classifying
    the entire radiograph as a joint.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics: pip install ultralytics") from exc
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")
    detector = YOLO(str(model_path))
    results = detector.predict(source=image, conf=confidence, imgsz=image_size, verbose=False)
    if not results or results[0].boxes is None:
        return []
    target = Path(output_dir) if output_dir else None
    if target:
        target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    h, w = image.shape[:2]
    names = results[0].names
    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = np.rint(box.xyxy[0].cpu().numpy()).astype(int)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = image[y1:y2, x1:x2].copy()
        class_id = int(box.cls.item())
        saved_path = None
        if target:
            saved_path = target / f"{image_path.stem}_roi_{idx:03d}.png"
            if not cv2.imwrite(str(saved_path), roi):
                raise OSError(f"Unable to save ROI to {saved_path}")
        records.append({"class_id": class_id, "class_name": names.get(class_id, str(class_id)),
                        "confidence": float(box.conf.item()), "bbox_xyxy": (x1, y1, x2, y2),
                        "image": roi, "path": saved_path})
    return records
