import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from backend.app import main


class StudyRouteTest(unittest.TestCase):
    def test_study_list_includes_modalities(self) -> None:
        response = MagicMock()
        response.json.return_value = ["study-xray", "study-ct"]
        with (
            patch.object(main, "orthanc_get", new=AsyncMock(return_value=response)),
            patch.object(
                main,
                "study_modalities",
                new=AsyncMock(side_effect=[{"CR"}, {"CT"}]),
            ),
        ):
            studies = asyncio.run(main.list_studies())

        self.assertEqual(
            studies,
            [
                {"id": "study-xray", "modalities": ["CR"]},
                {"id": "study-ct", "modalities": ["CT"]},
            ],
        )

    def test_classifier_rejects_ct_studies(self) -> None:
        with (
            patch.object(main, "classifier", object()),
            patch.object(
                main, "study_modalities", new=AsyncMock(return_value={"CT"})
            ),
            patch.object(main, "first_instance_id", new=AsyncMock()) as first_instance,
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.analyze_study("study-ct"))

        self.assertEqual(raised.exception.status_code, 422)
        first_instance.assert_not_awaited()

    def test_ct_segmentation_rejects_non_ct_studies(self) -> None:
        with (
            patch.object(main, "segmenter", object()),
            patch.object(
                main, "study_modalities", new=AsyncMock(return_value={"CR"})
            ),
            patch.object(main, "ct_series_files", new=AsyncMock()) as get_ct_files,
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.segment_study("study-xray"))

        self.assertEqual(raised.exception.status_code, 422)
        get_ct_files.assert_not_awaited()

    def test_ct_analysis_is_not_returned_as_a_classifier_result(self) -> None:
        with (
            patch.object(
                main, "study_modalities", new=AsyncMock(return_value={"CT"})
            ),
            patch.object(main.audit, "get_latest_analysis") as get_latest,
        ):
            result = asyncio.run(main.latest_analysis("study-ct"))

        self.assertIsNone(result)
        get_latest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
