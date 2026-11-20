import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.app.audit import (
    get_segmentations,
    get_latest_analysis,
    get_reviews,
    get_automation_status,
    has_analysis,
    initialize,
    save_analysis,
    save_automation_status,
    save_review,
    save_segmentation,
)


class AuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "audit.sqlite3")
        initialize(self.database)
        self.analysis_id = save_analysis(
            self.database, "study-1", {"label": "example", "score": 0.2}
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_review_is_persisted_for_its_analysis(self) -> None:
        review = save_review(
            self.database, self.analysis_id, "study-1", "reviewer-1", "approve", "checked"
        )
        self.assertIsNotNone(review)
        self.assertEqual(len(get_reviews(self.database, "study-1")), 1)

    def test_analysis_cannot_be_reviewed_under_another_study(self) -> None:
        review = save_review(
            self.database,
            self.analysis_id,
            "different-study",
            "reviewer-1",
            "reject",
            "",
        )
        self.assertIsNone(review)
        self.assertEqual(get_reviews(self.database, "study-1"), [])

    def test_latest_analysis_is_available_for_the_viewer(self) -> None:
        latest = get_latest_analysis(self.database, "study-1")
        self.assertEqual(latest["analysis_id"], self.analysis_id)
        self.assertEqual(latest["label"], "example")

    def test_segmentation_summary_is_persisted_without_review_data(self) -> None:
        segmentation_id = save_segmentation(
            self.database, "study-1", {"model_version": "test", "voxel_counts": {"tumor": 4}}
        )
        saved = get_segmentations(self.database, "study-1")
        self.assertEqual(saved[0]["id"], segmentation_id)
        self.assertEqual(saved[0]["result"]["voxel_counts"]["tumor"], 4)

    def test_automation_status_tracks_a_study_and_analysis(self) -> None:
        save_automation_status(
            self.database,
            "study-1",
            3,
            "completed",
            analysis_id=self.analysis_id,
            segmentation_id="segmentation-1",
        )
        status = get_automation_status(self.database, "study-1")
        self.assertEqual(status["instance_count"], 3)
        self.assertEqual(status["analysis_id"], self.analysis_id)
        self.assertEqual(status["segmentation_id"], "segmentation-1")
        self.assertTrue(has_analysis(self.database, "study-1"))

    def test_initialize_migrates_existing_automation_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = str(Path(temporary_directory) / "legacy.sqlite3")
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "CREATE TABLE study_automation (study_id TEXT PRIMARY KEY, "
                    "instance_count INTEGER NOT NULL, status TEXT NOT NULL, "
                    "analysis_id TEXT, error TEXT, updated_at TEXT NOT NULL)"
                )
            finally:
                connection.close()
            initialize(database)
            connection = sqlite3.connect(database)
            try:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(study_automation)")
                }
            finally:
                connection.close()
            self.assertIn("segmentation_id", columns)


if __name__ == "__main__":
    unittest.main()
