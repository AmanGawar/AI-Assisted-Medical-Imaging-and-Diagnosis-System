# AI-Assisted Medical Imaging and Diagnosis System

An educational chest-imaging prototype for exploring DICOM ingestion, image
classification, Grad-CAM visualizations, and radiologist review workflows.

> **For education only — not for clinical use.** This project is not a medical
> device and must not be used to diagnose, screen, triage, or guide treatment.
> Model scores are uncalibrated research outputs, not probabilities. A
> qualified clinician must not rely on this prototype for patient care.

## What is included

- Orthanc-backed DICOM study listing and preview.
- FastAPI service with an optional DenseNet121 checkpoint inference path.
- Stable-study Orthanc polling that can automatically run classifier inference.
- Grad-CAM overlays and explicit handling when a model is not configured.
- Persistent SQLite analysis and approve/reject audit records.
- React/TypeScript review screen.
- RSNA Pneumonia Detection classifier and Medical Segmentation Decathlon MONAI 3D U-Net
  training and inference paths.
- DVC pipeline configuration and local Docker Compose setup.

Medical datasets and trained checkpoints are kept outside Git. The local
classifier checkpoint is mounted into the API when present. The 3D segmentation
button remains disabled until `models/segmentation.pt` is available.

## Run locally with Docker

Requirements: Docker Desktop with Compose. Python 3.11+ and the project
environment are needed for training; training uses CUDA when available.

1. Copy `.env.example` to `.env`. The example credentials and local-only port
   bindings are for development; replace them before any shared deployment.
2. Start the services:

   ```powershell
   docker compose up --build
   ```

3. Open the React app at <http://localhost:5173>. Orthanc is at
   <http://localhost:8042> (development credentials are in `.env.example`).
4. Upload a permitted DICOM study through Orthanc. Once the study has been
   unchanged for 15 seconds, the API routes CR/DX studies to the classifier and
   CT studies to 3D segmentation when the matching checkpoint is present. The
   UI also supports manual inference. Segmentation requires at least eight CT
   slices and `models/segmentation.pt`.

The API is at <http://localhost:8000/docs>; automation state is available at
`GET /api/automation`. By default all ports are bound to loopback. Do not expose
the services to a network: this prototype has no application authentication or
authorization. See [the API reference](docs/api.md) and
[dataset notes](docs/data.md).

## Train the classifier

The classifier uses the RSNA 2018 Pneumonia Detection Challenge's original
image ZIP and its matching initial annotations JSON. The app expects the
annotations JSON at `data/rsna/annotations.json`; the image archive can remain
in Downloads. The data loader reads DICOMs directly from the ZIP without
extracting them. Do not commit the archive or annotations.

The RSNA page lists the [image archive](https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-original_2018.zip)
as 3.3 GB and separately provides the matching [initial annotations JSON](https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-annotations-original_2018.json).
Review the [dataset description and attribution terms](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
before using or sharing results. The training target is the dataset's
`Lung Opacity` versus `No Lung Opacity` annotation, not a general pneumonia
diagnosis.
The downloaded archive and annotations matched 25,684 DICOM studies in this
workspace: 5,659 `Lung Opacity` and 20,025 `No Lung Opacity` labels.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# Optional for an NVIDIA GPU on Windows; use this instead of CPU-only PyTorch:
.\.venv\Scripts\python.exe -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
.\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
New-Item -ItemType Directory -Force data\rsna
Invoke-WebRequest `
  -Uri "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-annotations-original_2018.json" `
  -OutFile data\rsna\annotations.json
.\.venv\Scripts\python.exe ml\train_classifier.py `
  --archive "$HOME\Downloads\pneumonia-challenge-dataset-original_2018.zip" `
  --annotations data\rsna\annotations.json `
  --output models\classifier.pt `
  --epochs 2 `
  --max-samples 2000 `
  --metrics-output reports\classifier_metrics.json
```

The default 2,000-study subset is stratified and split by study into 80% train
and 20% validation. Pass `--max-samples 0` to use all labeled studies. The
script initializes DenseNet121 with ImageNet weights; pass `--no-pretrained`
to skip that download. It prints validation BCE, ROC-AUC, and accuracy at a
0.5 threshold. These metrics describe only the held-out split from this
dataset and are not clinical performance estimates. The checkpoint's
`Lung Opacity` output label is compatible with the API's Grad-CAM path.

### Initial experiment

The latest local run used seed 42, 2,000 stratified studies (1,600 train / 400
validation), ImageNet-pretrained DenseNet121, batch size 8, and two epochs on
an RTX 3050. Validation BCE was **0.5583**, ROC-AUC **0.8430**, accuracy at
threshold 0.5 **0.7350**, precision **0.4444**, recall **0.8182**, and F1
**0.5760**. The confusion matrix was TN=222, FP=90, FN=16, TP=72. The complete
report is `reports/classifier_metrics.json`; re-run training to regenerate it.
This small, single-dataset experiment is only a baseline; it does not
establish clinical utility or generalization.

## Train the segmentation baseline

The official Medical Segmentation Decathlon Task06 Lung archive is about 9.2 GB
and is licensed CC-BY-SA 4.0. Download and extract it locally:

```powershell
New-Item -ItemType Directory -Force data\msd | Out-Null
curl.exe -L --continue-at - `
  -o data\msd\Task06_Lung.tar `
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar
tar -xf data\msd\Task06_Lung.tar -C data\msd
```

Train a deterministic subset for an educational baseline (the full task has
63 labeled training volumes):

```powershell
.\.venv\Scripts\python.exe ml\train_segmentation.py `
  --data-dir data\msd\Task06_Lung `
  --output models\segmentation.pt `
  --metrics-output reports\segmentation_metrics.json `
  --epochs 10 `
  --max-cases 16 `
  --roi-size 64
```

The script holds out cases before training and records per-class foreground
Dice in `reports/segmentation_metrics.json`. The API sorts and validates a CT
DICOM series, resamples it to the training spacing, and displays a central
model-space slice with the predicted mask. It also stores a summary record in
SQLite. This is a small internal holdout, not the official MSD test set.

The current 16-case, 10-epoch run used 12 training and 4 validation volumes.
Its cancer-class foreground Dice was **0.00024**, effectively no useful
segmentation on this split. The pipeline and viewer integration run, but this
checkpoint is a failed research baseline, not evidence of segmentation
performance. The official MSD test labels are unavailable here.

To try the CT segmentation button in the UI, convert one local MSD NIfTI
training volume to a new educational DICOM series and import it into Orthanc:

```powershell
$volume = Get-ChildItem data\msd\Task06_Lung\imagesTr -Filter "*.nii*" |
  Where-Object { $_.Name -notlike "._*" } |
  Select-Object -First 1
.\.venv\Scripts\python.exe ml\import_msd_demo_series.py --image $volume.FullName
```

The helper writes fresh DICOM identifiers and a non-identifying demo patient
label. It converts the volume for local viewer compatibility; it does not
reconstruct or preserve source DICOM metadata.

## DVC and data handling

`dvc.yaml` defines classifier and segmentation training stages, including
evaluation reports. To track the local data, copy the RSNA archive into the
workspace and add both datasets to DVC:

```powershell
Copy-Item "$HOME\Downloads\pneumonia-challenge-dataset-original_2018.zip" data\rsna\images.zip
$env:Path = (Resolve-Path .venv\Scripts).Path + ";" + $env:Path
.\.venv\Scripts\python.exe -m dvc add -f data\rsna\images.zip
.\.venv\Scripts\python.exe -m dvc add -f data\rsna\annotations.json
.\.venv\Scripts\python.exe -m dvc add -f data\msd\Task06_Lung
.\.venv\Scripts\python.exe -m dvc repro
```

The path update makes DVC's training subprocesses use the project environment,
not another system Python. On other platforms, activate `.venv` before running
`dvc repro`. This tracks data versions locally; configure a private DVC remote
before sharing or restoring data on another machine:

```powershell
.\.venv\Scripts\python.exe -m dvc remote add -d storage <your-private-DVC-remote>
.\.venv\Scripts\python.exe -m dvc push
```

Confirm the dataset's license and institutional policy permit storing it in
that remote. Never use public remotes for identifiable or restricted data.

## API outline

- `GET /api/health` — service and model readiness.
- `GET /api/automation` — auto-analysis configuration and per-study status.
- `GET /api/studies` — Orthanc study identifiers only (no patient metadata).
- `GET /api/studies/{study_id}/preview` — preview image of the first instance.
- `POST /api/studies/{study_id}/analyze` — inference and Grad-CAM, if a
  checkpoint is configured.
- `POST /api/studies/{study_id}/segment` — MONAI 3D segmentation for a CT series.
- `GET /api/segmentations?study_id=...` — stored segmentation summaries.
- `POST /api/studies/{study_id}/review` — append an approve/reject decision
  linked to an analysis and reviewer identifier.
- `GET /api/audit?study_id=...` — review history.

Audit records are append-only through the API, but the prototype does not
provide tamper-evident logging, user identity verification, or regulatory
controls.

## Architecture

See the [architecture diagram and trust boundaries](docs/architecture.md).

## Presentation and reflections

After both evaluation reports are available, regenerate the 10-minute deck:

```powershell
.\.venv\Scripts\python.exe presentation\build_deck.py
```

The generated slides are in `presentation/medical-imaging-demo.pptx`; the
speaker guide is `presentation/talk-track.md`. Each intern should complete
their own `presentation/reflection-template.md` with their actual contribution
and learning.

## Limitations, bias, and safe use

- The RSNA baseline labels radiographic opacity, not a clinical pneumonia
  diagnosis. Labels and boxes reflect the source cohort and annotation process.
  Results may not generalize across hospitals, devices, demographics, or
  workflows.
- The displayed sigmoid score is not calibrated and is not a measure of
  diagnostic certainty. Grad-CAM is a coarse visualization, not proof of
  causal reasoning or localization accuracy.
- The segmentation action supports only axial CT series with at least eight
  slices. It shows a central slice after resampling; it does not provide a
  clinical-grade 3D viewer or validated segmentation.
- Classification processes only the first instance of a study; it does not
  handle multi-view aggregation, prior studies, or clinical context.
- Segmentation selects the largest CT series but does not support arbitrary
  modalities, mixed geometries, or all DICOM transfer syntaxes.
- Automatic model routing waits for a stable study and records errors in
  `/api/automation`; it is not a production queue or clinical workflow system.
- Validate data rights, privacy, fairness, performance, security, and workflow
  safety before any use beyond controlled education or research.

## Submission checklist

- The app and architecture/API/setup notes are in this repository.
- Regenerate both evaluation reports after training before presenting results.
- Use `presentation/medical-imaging-demo.pptx` as the 10-minute presentation.
- Each intern must complete their own section in
  `presentation/reflection-template.md`; do not submit the blank template as a
  personal reflection.
- Do not present the prototype as clinically validated.
