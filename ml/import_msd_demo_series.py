import argparse
import base64
import io
import os
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def make_ct_instance(
    pixels: np.ndarray,
    image_position_lps: np.ndarray,
    orientation_lps: np.ndarray,
    pixel_spacing: tuple[float, float],
    slice_spacing: float,
    instance_number: int,
    study_uid: str,
    series_uid: str,
) -> bytes:
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.PatientName = "MSD^EDUCATIONAL"
    dataset.PatientID = "MSD-Task06-demo"
    dataset.StudyDescription = "Educational MSD Task06 sample"
    dataset.SeriesDescription = "Converted from public MSD NIfTI"
    dataset.InstanceNumber = instance_number
    dataset.ImagePositionPatient = [float(value) for value in image_position_lps]
    dataset.ImageOrientationPatient = [float(value) for value in orientation_lps]
    dataset.PixelSpacing = [float(value) for value in pixel_spacing]
    dataset.SliceThickness = float(slice_spacing)
    dataset.SpacingBetweenSlices = float(slice_spacing)
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    scaled_pixels = np.clip(np.rint(pixels), -32768, 32767).astype("<i2")
    dataset.PixelData = scaled_pixels.tobytes()
    output = io.BytesIO()
    pydicom.dcmwrite(output, dataset, enforce_file_format=True)
    return output.getvalue()


def nifti_as_ct_series(image_path: Path) -> list[bytes]:
    image = nib.load(image_path)
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("The MSD demo importer expects a single 3D NIfTI volume")
    affine_ras = np.asarray(image.affine, dtype=np.float64)
    axis_vectors = affine_ras[:3, :3]
    spacing = np.linalg.norm(axis_vectors, axis=0)
    if np.any(spacing <= 0):
        raise ValueError("The NIfTI volume has invalid voxel spacing")
    directions = axis_vectors / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-3):
        raise ValueError("The NIfTI affine is sheared and cannot be represented by this DICOM demo")

    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = ras_to_lps @ axis_vectors
    affine_lps[:3, 3] = ras_to_lps @ affine_ras[:3, 3]
    orientation = np.concatenate((directions[:, 1], directions[:, 0]))
    orientation[:3] = ras_to_lps @ orientation[:3]
    orientation[3:] = ras_to_lps @ orientation[3:]

    study_uid = generate_uid()
    series_uid = generate_uid()
    return [
        make_ct_instance(
            pixels=volume[:, :, slice_index],
            image_position_lps=(affine_lps @ np.array([0, 0, slice_index, 1]))[:3],
            orientation_lps=orientation,
            pixel_spacing=(float(spacing[0]), float(spacing[1])),
            slice_spacing=float(spacing[2]),
            instance_number=slice_index + 1,
            study_uid=study_uid,
            series_uid=series_uid,
        )
        for slice_index in range(volume.shape[2])
    ]


def upload_series(instances: list[bytes], url: str, username: str, password: str) -> None:
    authorization = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    for contents in instances:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/instances",
            data=contents,
            headers={
                "Authorization": f"Basic {authorization}",
                "Content-Type": "application/dicom",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status not in (200, 201):
                raise RuntimeError(f"Orthanc returned HTTP {response.status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one de-identified MSD NIfTI volume to an educational DICOM series"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--orthanc-url", default=os.getenv("ORTHANC_URL", "http://localhost:8042"))
    parser.add_argument("--username", default=os.getenv("ORTHANC_USERNAME", "demo"))
    parser.add_argument(
        "--password", default=os.getenv("ORTHANC_PASSWORD", "demo-only-change-me")
    )
    args = parser.parse_args()
    instances = nifti_as_ct_series(args.image)
    upload_series(instances, args.orthanc_url, args.username, args.password)
    print(f"Imported {len(instances)} educational CT instances into Orthanc.")


if __name__ == "__main__":
    main()
