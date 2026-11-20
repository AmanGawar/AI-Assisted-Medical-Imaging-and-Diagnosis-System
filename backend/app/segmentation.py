import base64
import io
import tempfile
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pydicom
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from PIL import Image


def build_unet(out_channels: int) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
    )


def dicom_series_to_nifti(instances: Sequence[bytes]) -> nib.Nifti1Image:
    if len(instances) < 8:
        raise ValueError("3D segmentation requires at least 8 CT slices in one series")

    slices = [pydicom.dcmread(io.BytesIO(contents)) for contents in instances]
    if any(getattr(dataset, "Modality", "") != "CT" for dataset in slices):
        raise ValueError("3D segmentation currently supports CT series only")

    first = slices[0]
    required_tags = (
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "ImageOrientationPatient",
        "PixelSpacing",
        "ImagePositionPatient",
    )
    if any(not hasattr(first, tag) for tag in required_tags):
        raise ValueError("CT series is missing required geometry tags")
    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    if orientation.shape != (6,):
        raise ValueError("CT series has an invalid ImageOrientationPatient")
    column_direction = orientation[:3]
    row_direction = orientation[3:]
    normal = np.cross(column_direction, row_direction)
    normal_length = np.linalg.norm(normal)
    if normal_length <= 0:
        raise ValueError("CT series has invalid slice orientation")
    normal /= normal_length
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    if pixel_spacing.shape != (2,) or np.any(pixel_spacing <= 0):
        raise ValueError("CT series has invalid PixelSpacing")

    decoded_slices: list[tuple[float, np.ndarray]] = []
    for dataset in slices:
        if any(not hasattr(dataset, tag) for tag in required_tags):
            raise ValueError("CT series is missing required geometry tags")
        if dataset.StudyInstanceUID != first.StudyInstanceUID:
            raise ValueError("DICOM inputs contain more than one study")
        if dataset.SeriesInstanceUID != first.SeriesInstanceUID:
            raise ValueError("DICOM inputs contain more than one series")
        if not np.allclose(dataset.ImageOrientationPatient, orientation, atol=1e-4):
            raise ValueError("CT series contains inconsistent slice orientations")
        if not np.allclose(dataset.PixelSpacing, pixel_spacing, atol=1e-4):
            raise ValueError("CT series contains inconsistent pixel spacing")
        position = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("CT series has an invalid ImagePositionPatient")
        projection = float(np.dot(position, normal))
        try:
            pixels = dataset.pixel_array.astype(np.float32)
        except (AttributeError, NotImplementedError, ValueError) as exc:
            raise ValueError("A CT slice could not be decoded") from exc
        if pixels.ndim != 2 or pixels.shape != (int(dataset.Rows), int(dataset.Columns)):
            raise ValueError("CT series contains an unsupported pixel array")
        pixels *= float(getattr(dataset, "RescaleSlope", 1.0))
        pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
        decoded_slices.append((projection, pixels))

    decoded_slices.sort(key=lambda item: item[0])
    gaps = np.diff([projection for projection, _ in decoded_slices])
    if np.any(gaps <= 0):
        raise ValueError("CT series contains duplicate or unordered slice positions")
    slice_spacing = float(np.median(gaps))
    if np.max(np.abs(gaps - slice_spacing)) > max(0.5, slice_spacing * 0.1):
        raise ValueError("CT series has irregular slice spacing")
    if any(image.shape != decoded_slices[0][1].shape for _, image in decoded_slices):
        raise ValueError("CT series contains inconsistent image dimensions")

    volume = np.stack([image for _, image in decoded_slices], axis=-1)
    first_position = np.asarray(
        sorted(
            (np.asarray(dataset.ImagePositionPatient, dtype=np.float64) for dataset in slices),
            key=lambda position: float(np.dot(position, normal)),
        )[0],
        dtype=np.float64,
    )
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = row_direction * pixel_spacing[0]
    affine_lps[:3, 1] = column_direction * pixel_spacing[1]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = first_position
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return nib.Nifti1Image(volume, lps_to_ras @ affine_lps)


class LungVolumeSegmenter:
    def __init__(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.labels = {int(key): str(value) for key, value in checkpoint["labels"].items()}
        if sorted(self.labels) != list(range(len(self.labels))) or len(self.labels) < 2:
            raise ValueError("Segmentation checkpoint labels must be contiguous and include foreground")
        self.roi_size = tuple(int(value) for value in checkpoint.get("roi_size", (64, 64, 64)))
        self.model = build_unet(len(self.labels))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.version = str(checkpoint.get("model_version", Path(checkpoint_path).name))
        self.preprocess = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys="image", axcodes="RAS"),
                Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-1000,
                    a_max=400,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )

    def predict(self, instances: Sequence[bytes]) -> dict[str, Any]:
        nifti_volume = dicom_series_to_nifti(instances)
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "study.nii.gz"
            nib.save(nifti_volume, image_path)
            image = self.preprocess({"image": str(image_path)})["image"].unsqueeze(0)
            with torch.inference_mode():
                logits = sliding_window_inference(
                    image,
                    roi_size=self.roi_size,
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=0.25,
                )
                prediction = logits.argmax(dim=1)[0].cpu().numpy()

            model_volume = image[0, 0].cpu().numpy()
            slice_index = model_volume.shape[-1] // 2
            image_slice = np.clip(model_volume[..., slice_index], 0.0, 1.0)
            mask_slice = prediction[..., slice_index]
            output = self._render_overlay(image_slice, mask_slice)
            voxel_counts = {
                self.labels[label]: int(np.count_nonzero(prediction == label))
                for label in range(1, len(self.labels))
            }

        return {
            "overlay_png_base64": base64.b64encode(output).decode("ascii"),
            "slice_index": slice_index,
            "slice_count": int(model_volume.shape[-1]),
            "voxel_counts": voxel_counts,
            "model_version": self.version,
            "disclaimer": "For education only, not for clinical use.",
        }

    @staticmethod
    def _render_overlay(image: np.ndarray, mask: np.ndarray) -> bytes:
        grayscale = (image * 255).astype(np.uint8)
        rendered = np.repeat(grayscale[:, :, None], 3, axis=2).astype(np.float32)
        colors = ((255, 70, 50), (30, 230, 140), (60, 150, 255), (245, 210, 45))
        for label in np.unique(mask):
            if label == 0:
                continue
            color = np.asarray(colors[(int(label) - 1) % len(colors)], dtype=np.float32)
            selected = mask == label
            rendered[selected] = rendered[selected] * 0.55 + color * 0.45
        output = io.BytesIO()
        Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8), mode="RGB").save(
            output, format="PNG"
        )
        return output.getvalue()
