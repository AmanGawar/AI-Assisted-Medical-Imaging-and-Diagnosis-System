# Architecture

```mermaid
flowchart LR
  U[Permitted DICOM upload] --> O[Orthanc DICOM server]
  O -->|Study and instance APIs| A[FastAPI service]
  O -->|Stable-study polling| A
  A --> C[DenseNet121 classifier + Grad-CAM]
  A --> M[MONAI 3D U-Net]
  A --> DB[(SQLite analyses, segmentations, reviews, automation status)]
  A --> UI[React review and visualization UI]
  UI -->|Manual inference and review decisions| A
  R[RSNA X-ray data] --> DVC[DVC-tracked training pipeline]
  S[MSD Task06 CT volumes] --> DVC
  DVC --> C
  DVC --> M
```

The local demo runs Orthanc, FastAPI, and the React app as separate Compose
services. Ports are bound to loopback. CR/DX classifier inference processes
the first instance; CT segmentation selects the largest CT series in a study,
sorts its slices by physical position, resamples the volume, and displays a
central model-space slice. The SQLite database stores output summaries and
review records, not patient metadata or image files.

## Trust and data boundaries

- DICOM files remain in the local Orthanc storage volume.
- Model training inputs and checkpoints are excluded from Git; DVC stores data
  in its local cache until an approved private remote is configured.
- The browser calls the local API. The API accesses Orthanc using local
  development credentials.
- This prototype has no user authentication, authorization, tamper-evident
  audit controls, or clinical safety controls.
