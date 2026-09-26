"""Automated feature extractors, ordinal scoring heads, and Grad-CAM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def _load_densenet(pretrained: bool) -> nn.Module:
    try:
        return densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
    except (TypeError, AttributeError):  # Compatibility with older torchvision.
        return densenet121(pretrained=pretrained)


def load_checkpoint(model: nn.Module, checkpoint: str | Path, device: torch.device) -> nn.Module:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


class DenseNetBinaryClassifier(nn.Module):
    """DenseNet121 automated feature extractor with a binary spatial classifier."""
    def __init__(self, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        self.backbone = _load_densenet(pretrained)
        features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.backbone(x))).squeeze(1)


class _OrdinalHead(nn.Module):
    """Cumulative-link ordinal head with monotonically ordered cut points."""
    def __init__(self, in_features: int, grades: int):
        super().__init__()
        self.score = nn.Linear(in_features, 1)
        self.first_threshold = nn.Parameter(torch.tensor(0.0))
        self.threshold_deltas = nn.Parameter(torch.zeros(grades - 2))

    def thresholds(self) -> torch.Tensor:
        if self.threshold_deltas.numel() == 0:
            return self.first_threshold.reshape(1)
        increments = torch.nn.functional.softplus(self.threshold_deltas) + 1e-4
        return torch.cat((self.first_threshold.reshape(1), self.first_threshold + torch.cumsum(increments, 0)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Output j estimates P(grade > j), suitable for BCEWithLogitsLoss.
        return self.score(features) - self.thresholds().unsqueeze(0)


class OrdinalSeverityClassifier(nn.Module):
    """Predict JSN grades 0–4 and erosion grades 0–5 from an ROI tensor."""
    def __init__(self, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        self.backbone = _load_densenet(pretrained)
        features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.jsn_head = _OrdinalHead(features, grades=5)
        self.erosion_head = _OrdinalHead(features, grades=6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        return {"jsn_logits": self.jsn_head(features), "erosion_logits": self.erosion_head(features)}


def ordinal_grade(logits: torch.Tensor) -> torch.Tensor:
    """Decode cumulative ordinal logits as the number of exceeded cut points."""
    return (logits >= 0).sum(dim=-1)


class GradCAM:
    """Grad-CAM targeting DenseNet's last convolutional feature map."""
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise TypeError("GradCAM expects a model with a DenseNet 'backbone'")
        self.target_layer = target_layer or backbone.features.denseblock4
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = [self.target_layer.register_forward_hook(self._save_activation),
                       self.target_layer.register_full_backward_hook(self._save_gradient)]

    def _save_activation(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(self, _module: nn.Module, _grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, target: int | None = None,
                 output_size: tuple[int, int] | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        if isinstance(output, dict):
            logits = output["jsn_logits"]
            target_score = logits[:, 0].sum() if target is None else logits[:, target].sum()
        else:
            target_score = output.sum() if target is None else output[:, target].sum()
        target_score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        maximum = float(cam.max())
        return (cam / maximum if maximum > 0 else cam).astype(np.float32)

    @staticmethod
    def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
        return cv2.addWeighted(image.astype(np.uint8), 1 - alpha, colored, alpha, 0)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def __del__(self) -> None:
        self.close()
