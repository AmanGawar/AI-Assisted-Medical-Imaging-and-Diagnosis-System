import base64
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from backend.app.segmentation import (
    LungVolumeSegmenter,
    build_unet,
    dicom_series_to_nifti,
)

STUDY_UID = generate_uid()
SERIES_UID = generate_uid()


def make_ct_slice(position: int) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = STUDY_UID
    dataset.SeriesInstanceUID = SERIES_UID
    dataset.Modality = "CT"
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.PixelSpacing = [0.7, 0.8]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, position * 2]
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = -1000
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.full((4, 5), position + 1000, dtype=np.uint16).tobytes()
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, dataset, enforce_file_format=True)
    return stream.getvalue()


class SegmentationSeriesTest(unittest.TestCase):
    def test_ct_slices_are_sorted_and_rescaled_into_a_3d_volume(self) -> None:
        contents = [make_ct_slice(position) for position in reversed(range(8))]
        volume = dicom_series_to_nifti(contents)

        self.assertEqual(volume.shape, (4, 5, 8))
        np.testing.assert_array_equal(volume.get_fdata()[:, :, 0], np.zeros((4, 5)))
        np.testing.assert_array_equal(volume.get_fdata()[:, :, 7], np.full((4, 5), 7))
        self.assertAlmostEqual(float(np.linalg.norm(volume.affine[:3, 0])), 0.7)
        self.assertAlmostEqual(float(np.linalg.norm(volume.affine[:3, 1])), 0.8)
        self.assertAlmostEqual(float(np.linalg.norm(volume.affine[:3, 2])), 2.0)

    def test_single_slice_is_rejected_with_a_clear_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 8 CT slices"):
            dicom_series_to_nifti([make_ct_slice(0)])

    def test_checkpoint_runs_a_3d_inference_and_returns_a_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "segmentation.pt"
            model = build_unet(2)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": {0: "background", 1: "lesion"},
                    "model_version": "test",
                    "roi_size": (32, 32, 32),
                },
                checkpoint_path,
            )
            segmenter = LungVolumeSegmenter(str(checkpoint_path))
            result = segmenter.predict([make_ct_slice(index) for index in range(8)])

        png = base64.b64decode(result["overlay_png_base64"])
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(result["model_version"], "test")
        self.assertGreater(result["slice_count"], 0)


if __name__ == "__main__":
    unittest.main()
