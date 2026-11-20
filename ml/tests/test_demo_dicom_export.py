import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from backend.app.segmentation import dicom_series_to_nifti
from ml.import_msd_demo_series import nifti_as_ct_series


class DemoDicomExportTest(unittest.TestCase):
    def test_nifti_volume_round_trips_as_a_geometry_preserving_ct_series(self) -> None:
        source_data = np.arange(4 * 5 * 8, dtype=np.int16).reshape((4, 5, 8))
        affine = np.diag([0.7, 0.8, 2.0, 1.0])
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "volume.nii.gz"
            nib.save(nib.Nifti1Image(source_data, affine), image_path)
            instances = nifti_as_ct_series(image_path)

        self.assertEqual(len(instances), 8)
        converted = dicom_series_to_nifti(instances)
        canonical = nib.as_closest_canonical(converted)
        np.testing.assert_array_equal(canonical.get_fdata(), source_data)
        np.testing.assert_allclose(canonical.affine, affine, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
