# 10-minute presentation talk track

Use the figures on the slides generated from the latest evaluation reports.
This outline intentionally avoids claiming clinical performance.

| Slide | Time | Speaker notes |
|---|---:|---|
| 1. Title | 0:30 | Introduce the project as an educational prototype. Read the clinical-use disclaimer aloud. |
| 2. Problem and goal | 1:00 | Explain the workflow being demonstrated: image ingestion, model-assisted review, and auditable human decisions. Emphasize that the output is not a diagnosis. |
| 3. Architecture | 1:15 | Walk through Orthanc, stable-study processing, FastAPI inference, the React review screen, and SQLite records. Explain that all services are local and data are not committed to Git. |
| 4. Data and models | 1:15 | Identify the RSNA binary opacity label and MSD Task06 CT segmentation task. Describe the data split and DVC approach; do not imply the two datasets are interchangeable. |
| 5. Classifier results | 1:15 | Read the measured metrics and confusion matrix from the slide. Explain the threshold and same-source held-out split. Call out the limited sample and uncalibrated scores. |
| 6. Segmentation results | 1:15 | Report the internal split and near-zero cancer foreground Dice (0.00024). State plainly that this run failed to demonstrate useful segmentation; the official test labels are unavailable and this is not clinical validation. |
| 7. Demo walkthrough | 1:15 | Show study selection, manual or automatic classification, Grad-CAM, CT segmentation when available, and the review/audit flow. |
| 8. Limits and next steps | 1:45 | Discuss bias, domain shift, missing clinical context, security, and why model outputs cannot be used for care. Identify the next validation and deployment work, then state each intern's actual ownership. |
| Questions / transition | 0:30 | Invite questions and confirm which intern owns each component. |

Total: approximately 10 minutes.
