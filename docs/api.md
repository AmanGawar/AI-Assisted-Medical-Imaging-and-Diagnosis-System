# API reference

The local API is documented interactively at <http://localhost:8000/docs>.
All endpoints are intended for loopback-only educational use. The API does not
authenticate users or authorize clinical access.

## Readiness and automation

### `GET /api/health`

Returns service and model readiness:

```json
{
  "status": "ok",
  "model_ready": true,
  "segmentation_ready": false,
  "auto_analysis": true,
  "disclaimer": "For education only, not for clinical use."
}
```

`model_ready` and `segmentation_ready` reflect checkpoints loaded at API
startup. Mount a checkpoint under `models/` and restart the API to load it.

### `GET /api/automation`

Returns whether stable-study polling is enabled, its timing, and per-study
status (`processing`, `completed`, or `failed`). Errors are retained in the
status record and written to the API log. Once the Orthanc instance count has
remained unchanged for the configured stability period, CR/DX studies are
routed to the classifier and CT studies to 3D segmentation. Unsupported
modalities are reported as failed rather than sent to an incompatible model.

## Study access and classification

### `GET /api/studies`

Returns Orthanc study identifiers and their series modalities, without patient
metadata. The viewer enables DenseNet classification only for CR/DX studies
and 3D segmentation only for CT studies. The API also rejects incompatible
manual inference requests.

### `GET /api/studies/{study_id}/preview`

Returns a PNG preview of the first instance in a study.

### `POST /api/studies/{study_id}/analyze`

Runs the DenseNet classifier on the first instance and returns the predicted
RSNA class, an **uncalibrated** score, a Grad-CAM PNG encoded as base64, the
model version, and a persistent analysis identifier. The target is
`Lung Opacity` versus `No Lung Opacity`, not a pneumonia diagnosis. Only
studies containing CR/DX series are accepted; incompatible modalities receive
`422`.

### `GET /api/studies/{study_id}/analysis`

Returns the latest saved classifier result, including its overlay, or `null` if
the study has not been analyzed. The UI polls this endpoint to display
automatic results after the stable-study worker finishes.

## 3D segmentation

### `POST /api/studies/{study_id}/segment`

Selects the largest CT series in the study, requiring at least eight slices.
It sorts the DICOM slices by their physical positions, validates the series
geometry, converts the stack to a volume, resamples it to the training
spacing, and runs sliding-window MONAI 3D U-Net inference.

The response contains a base64 PNG of the central resampled model-space slice,
the model version, predicted foreground voxel counts, and a persistent
segmentation identifier. The PNG is a visualization of the model-space slice;
it is not a clinical-grade 3D viewer or a validated mask.

Common errors:

- `422`: no CT series, fewer than eight slices, or inconsistent
  DICOM geometry.
- `503`: the segmentation checkpoint is not configured.
- `502` / `503`: Orthanc returned an error or became unavailable while the
  series was read.

### `GET /api/segmentations?study_id={study_id}`

Returns stored summary records for segmentations of the requested study. The
large visualization image itself is not stored in SQLite.

## Human review and audit

### `POST /api/studies/{study_id}/review`

Requires a reviewer identifier, an analysis identifier, a decision, and an
optional note:

```json
{
  "analysis_id": "analysis UUID",
  "reviewer": "reviewer identifier",
  "decision": "approve",
  "note": "Optional educational review note"
}
```

Decisions are append-only in the prototype's SQLite database. This is not a
tamper-evident audit system and does not verify the reviewer's identity.

### `GET /api/audit?study_id={study_id}`

Returns the recorded approve/reject decisions for one study.

## Safety

Every model output and review function is for education only and must not be
used for patient care. The prototype has no authentication, authorization,
calibration, regulatory controls, or clinical validation.
