import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.models import DenseNet121_Weights, densenet121


class ChestXrayClassifier:
    def __init__(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.labels = checkpoint["labels"]
        if not isinstance(self.labels, list) or not self.labels:
            raise ValueError("Classifier checkpoint must contain a non-empty labels list")

        self.model = densenet121(weights=None)
        self.model.classifier = torch.nn.Linear(
            self.model.classifier.in_features, len(self.labels)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.version = str(checkpoint.get("model_version", Path(checkpoint_path).name))
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self.model.features.denseblock4.register_forward_hook(self._capture_activation)
        self.model.features.denseblock4.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, _module: Any, _inputs: Any, output: torch.Tensor) -> None:
        self._activations = output

    def _capture_gradient(
        self, _module: Any, _grad_input: Any, grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        self._gradients = grad_output[0]

    @staticmethod
    def _decode_dicom(contents: bytes) -> Image.Image:
        dataset = pydicom.dcmread(io.BytesIO(contents))
        pixels = dataset.pixel_array
        if pixels.ndim == 3:
            pixels = pixels[0]
        pixels = pixels.astype(np.float32)
        pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0))
        pixels = pixels + float(getattr(dataset, "RescaleIntercept", 0.0))
        low, high = np.percentile(pixels, [1, 99])
        if high <= low:
            high = low + 1.0
        pixels = np.clip((pixels - low) / (high - low), 0, 1)
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 1 - pixels
        grayscale = (pixels * 255).astype(np.uint8)
        return Image.fromarray(grayscale).convert("RGB")

    def predict(self, dicom_bytes: bytes) -> dict[str, Any]:
        image = self._decode_dicom(dicom_bytes).resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(tensor)
            scores = torch.sigmoid(logits)[0]
            selected = int(torch.argmax(scores).item())
            logits[0, selected].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM activations were not captured")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam / cam.max().clamp_min(1e-8)
        heat = cam.detach().cpu().numpy()

        overlay = np.zeros((224, 224, 4), dtype=np.uint8)
        overlay[:, :, 0] = 255
        overlay[:, :, 1] = (255 * (1 - heat)).astype(np.uint8)
        overlay[:, :, 2] = 0
        overlay[:, :, 3] = (150 * heat).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(overlay, mode="RGBA").save(output, format="PNG")

        return {
            "label": self.labels[selected],
            "score": float(scores[selected].detach().item()),
            "overlay_png_base64": base64.b64encode(output.getvalue()).decode("ascii"),
            "model_version": self.version,
        }
