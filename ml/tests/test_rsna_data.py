import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from ml.rsna_data import load_rsna_records


class RSNADataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.archive_path = self.root / "images.zip"
        self.annotations_path = self.root / "annotations.json"
        with ZipFile(self.archive_path, "w") as archive:
            archive.writestr("study-a/series-a/sop-a.dcm", b"sample")
            archive.writestr("study-b/series-b/sop-b.dcm", b"sample")
        project = {
            "labelGroups": [
                {
                    "labels": [
                        {"id": "negative", "name": "No Lung Opacity"},
                        {"id": "positive", "name": "Lung Opacity"},
                    ]
                }
            ],
            "datasets": [
                {
                    "studies": [
                        {"StudyInstanceUID": "study-a"},
                        {"StudyInstanceUID": "study-b"},
                    ],
                    "annotations": [
                        {
                            "StudyInstanceUID": "study-a",
                            "SOPInstanceUID": "sop-a",
                            "labelId": "negative",
                        },
                        {
                            "StudyInstanceUID": "study-b",
                            "SOPInstanceUID": "sop-b",
                            "labelId": "positive",
                        },
                    ],
                }
            ],
        }
        self.annotations_path.write_text(json.dumps(project), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_maps_dicom_by_sop_uid_and_aggregates_binary_labels(self) -> None:
        records = load_rsna_records(self.archive_path, self.annotations_path)
        self.assertEqual(
            [(record.study_uid, record.member, record.label) for record in records],
            [
                ("study-a", "study-a/series-a/sop-a.dcm", 0),
                ("study-b", "study-b/series-b/sop-b.dcm", 1),
            ],
        )

    def test_missing_dicom_file_is_reported(self) -> None:
        with ZipFile(self.archive_path, "w") as archive:
            archive.writestr("study-a/series-a/sop-a.dcm", b"sample")
        with self.assertRaisesRegex(ValueError, "No DICOM file"):
            load_rsna_records(self.archive_path, self.annotations_path)


if __name__ == "__main__":
    unittest.main()
