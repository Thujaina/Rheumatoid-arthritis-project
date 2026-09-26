"""Train the binary RA and ordinal severity models from labeled CSV manifests.

Binary CSV columns: image_path,label (0=normal, 1=RA).
Severity CSV columns: image_path,jsn_grade,erosion_grade (grades 0..4 and 0..5).
Paths may be absolute or relative to the CSV file's directory.

Examples:
  python train.py binary --csv data/binary.csv --output checkpoints/ra_binary.pt
  python train.py severity --csv data/severity.csv --output checkpoints/ra_severity.pt

For Colab/Kaggle, use GPU runtime and point --csv at the mounted dataset.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from models import DenseNetBinaryClassifier, OrdinalSeverityClassifier
from preprocess import IMAGE_SUFFIXES, _read_image, enhance_clahe, build_augmentation


class ManifestDataset(Dataset):
    def __init__(self, csv_path: Path, task: str, training: bool, image_size: int = 224):
        self.csv_path = csv_path.resolve()
        self.task = task
        required = {"image_path", "label"} if task == "binary" else {"image_path", "jsn_grade", "erosion_grade"}
        with self.csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"No records found in {self.csv_path}")
        self.samples: list[tuple[Path, tuple[int, ...]]] = []
        for line, row in enumerate(self.rows, start=2):
            path = Path(row["image_path"].strip()).expanduser()
            if not path.is_absolute():
                path = self.csv_path.parent / path
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                raise ValueError(f"Invalid or missing image at CSV line {line}: {path}")
            try:
                if task == "binary":
                    labels = (int(row["label"]),)
                    if labels[0] not in (0, 1):
                        raise ValueError("binary label must be 0 or 1")
                else:
                    labels = (int(row["jsn_grade"]), int(row["erosion_grade"]))
                    if not 0 <= labels[0] <= 4 or not 0 <= labels[1] <= 5:
                        raise ValueError("JSN must be 0..4 and erosion must be 0..5")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label at CSV line {line}: {exc}") from exc
            self.samples.append((path, labels))
        self.transform = build_augmentation(image_size, training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, labels = self.samples[index]
        image = cv2.cvtColor(enhance_clahe(_read_image(path)), cv2.COLOR_GRAY2RGB)
        array = self.transform(image=image)["image"].transpose(2, 0, 1).copy()
        return torch.from_numpy(array).float(), torch.tensor(labels, dtype=torch.long)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ordinal_targets(grades: torch.Tensor, count: int) -> torch.Tensor:
    # Each binary target asks whether grade exceeds threshold 0, 1, ..., count-1.
    thresholds = torch.arange(count, device=grades.device)
    return (grades.unsqueeze(1) > thresholds).float()


def split_dataset(dataset: ManifestDataset, task: str, val_fraction: float, seed: int):
    indices = np.arange(len(dataset))
    if task == "binary":
        stratify = np.asarray([labels[0] for _, labels in dataset.samples])
    else:
        # Stratify by paired labels when possible; fall back to JSN to avoid sparse-pair failures.
        paired = np.asarray([f"{a}:{b}" for _, (a, b) in dataset.samples])
        counts = {key: int((paired == key).sum()) for key in np.unique(paired)}
        stratify = paired if min(counts.values()) >= 2 else np.asarray([a for _, (a, _) in dataset.samples])
    try:
        train_idx, val_idx = train_test_split(indices, test_size=val_fraction,
                                               random_state=seed, stratify=stratify)
    except ValueError as exc:
        raise ValueError(f"Cannot make requested stratified split: {exc}. Add examples per class or adjust --val-fraction.") from exc
    return train_idx, val_idx


def save_checkpoint(model: nn.Module, output: Path, task: str, epoch: int,
                    validation_loss: float, args: argparse.Namespace) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "task": task, "epoch": epoch,
                "validation_loss": validation_loss, "image_size": args.image_size,
                "class_mapping": {"normal": 0, "pathological_ra": 1} if task == "binary" else None}, output)


def run_epoch(model: nn.Module, loader: DataLoader, task: str, device: torch.device,
              optimizer: torch.optim.Optimizer | None, positive_weight: torch.Tensor | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    loss_sum, n = 0.0, 0
    binary_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    ordinal_loss = nn.BCEWithLogitsLoss()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(images)
            if task == "binary":
                loss = binary_loss(prediction, labels[:, 0].float())
            else:
                jsn_targets = ordinal_targets(labels[:, 0], count=4)
                erosion_targets = ordinal_targets(labels[:, 1], count=5)
                loss = ordinal_loss(prediction["jsn_logits"], jsn_targets)
                loss = loss + ordinal_loss(prediction["erosion_logits"], erosion_targets)
            if training:
                loss.backward()
                optimizer.step()
        size = images.size(0)
        loss_sum += float(loss.detach()) * size
        n += size
    return loss_sum / max(n, 1)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")
    dataset = ManifestDataset(csv_path, args.task, training=False, image_size=args.image_size)
    train_idx, val_idx = split_dataset(dataset, args.task, args.val_fraction, args.seed)
    train_data = ManifestDataset(csv_path, args.task, training=True, image_size=args.image_size)
    workers = args.workers
    train_loader = DataLoader(Subset(train_data, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    model: nn.Module = (DenseNetBinaryClassifier(pretrained=not args.no_pretrained) if args.task == "binary"
                        else OrdinalSeverityClassifier(pretrained=not args.no_pretrained)).to(device)
    positive_weight = None
    if args.task == "binary" and args.balance_classes:
        labels = np.asarray([dataset.samples[i][1][0] for i in train_idx])
        negatives, positives = int((labels == 0).sum()), int((labels == 1).sum())
        if positives == 0 or negatives == 0:
            raise ValueError("Training split must contain both binary classes")
        positive_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, args.task, device, optimizer, positive_weight)
        val_loss = run_epoch(model, val_loader, args.task, device, None)
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(model, Path(args.output), args.task, epoch, val_loss, args)
    print(f"Best validation checkpoint saved to {Path(args.output).resolve()} (loss={best:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["binary", "severity"])
    parser.add_argument("--csv", required=True, help="CSV manifest for this training task")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from ImageNet weights")
    parser.add_argument("--balance-classes", action="store_true", help="Weight positive binary examples")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.val_fraction < 1:
        parser.error("epochs/batch-size must be positive and val-fraction must be between 0 and 1")
    args.task = args.task
    return args


if __name__ == "__main__":
    train(parse_args())
