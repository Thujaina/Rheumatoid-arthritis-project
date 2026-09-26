"""Image processing and dataset utilities for the RA radiograph pipeline.

Expected layout: normal_dir contains MURA normal images; pathological_dir contains
RA-positive images. DICOM support is optional and requires pydicom.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import pydicom
except ImportError:  # PNG/JPEG use remains available without pydicom.
    pydicom = None

try:
    import albumentations as A
except ImportError as exc:
    raise ImportError("Install albumentations to use the preprocessing pipeline") from exc


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}


def _read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        if pydicom is None:
            raise ImportError("DICOM input requires pydicom (pip install pydicom)")
        ds = pydicom.dcmread(str(path))
        pixels = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels
        pixels -= pixels.min()
        maximum = pixels.max()
        pixels = pixels / maximum * 255.0 if maximum > 0 else pixels
        return np.clip(pixels, 0, 255).astype(np.uint8)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def enhance_clahe(image: np.ndarray, clip_limit: float = 2.0,
                  tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to grayscale or color images, preserving the input color space."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image.astype(np.uint8))
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    bgr = image[:, :, :3].astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.dstack((result, image[:, :, 3])) if image.shape[2] == 4 else result


def build_augmentation(image_size: int = 224, training: bool = True) -> A.Compose:
    """Geometrically conservative augmentation for musculoskeletal radiographs."""
    transforms = [A.Resize(image_size, image_size)]
    if training:
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=7, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.45),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03),
                    rotate=(-3, 3), shear=(-2, 2), p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.12, p=0.35),
        ]
    transforms.append(A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0))
    return A.Compose(transforms)


class RadiographDataset(Dataset):
    """Binary image dataset; class 0=normal and class 1=pathological/RA."""
    def __init__(self, normal_dir: str | Path, pathological_dir: str | Path,
                 image_size: int = 224, training: bool = False):
        self.samples: list[tuple[Path, int]] = []
        for root, label in ((Path(normal_dir), 0), (Path(pathological_dir), 1)):
            if not root.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {root}")
            found = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not found:
                raise ValueError(f"No supported images found under {root}")
            self.samples.extend((p, label) for p in found)
        self.transform = build_augmentation(image_size, training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = enhance_clahe(_read_image(path))
        # DenseNet uses three-channel input; replicate grayscale X-rays consistently.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        transformed = self.transform(image=image)
        tensor = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float()
        return tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(normal_dir: str | Path, pathological_dir: str | Path,
                       batch_size: int = 32, val_fraction: float = 0.2,
                       random_state: int = 42, num_workers: Optional[int] = None,
                       image_size: int = 224) -> tuple[DataLoader, DataLoader]:
    """Create stratified train/validation loaders; validation has no augmentation."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    base = RadiographDataset(normal_dir, pathological_dir, image_size=image_size, training=False)
    labels = np.asarray([label for _, label in base.samples])
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction,
                                           random_state=random_state, stratify=labels)
    train_data = RadiographDataset(normal_dir, pathological_dir, image_size=image_size, training=True)
    workers = num_workers if num_workers is not None else min(8, os.cpu_count() or 1)
    pin = torch.cuda.is_available()
    return (
        DataLoader(Subset(train_data, train_idx), batch_size=batch_size, shuffle=True,
                   num_workers=workers, pin_memory=pin, persistent_workers=workers > 0),
        DataLoader(Subset(base, val_idx), batch_size=batch_size, shuffle=False,
                   num_workers=workers, pin_memory=pin, persistent_workers=workers > 0),
    )
