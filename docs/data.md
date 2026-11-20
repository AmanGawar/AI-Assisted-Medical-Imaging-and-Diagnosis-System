# Dataset and evaluation notes

## RSNA Pneumonia Detection Challenge

- **Use:** binary chest-radiograph classification of `Lung Opacity` versus
  `No Lung Opacity`.
- **Local source:** original RSNA image archive and the matching initial
  annotations JSON.
- **Workspace inventory:** 25,684 matched studies; 5,659 positive and 20,025
  negative annotations.
- **Baseline training run:** seed 42, stratified 2,000-study subset, 1,600
  training and 400 validation studies, ImageNet-pretrained DenseNet121, batch
  size 8, two epochs.
- **Evaluation:** the held-out split comes from the same source data. See
  `reports/classifier_metrics.json` for the latest run's confusion matrix,
  ROC-AUC, accuracy, precision, recall, and F1 at a fixed 0.5 threshold.
- **Limit:** these are preliminary research results, not estimates of clinical
  performance. The source label is radiographic opacity, not a diagnosis of
  pneumonia.

The dataset is not included in Git. Review the [RSNA challenge data terms](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
before redistributing or publishing data or derived material.

## Medical Segmentation Decathlon — Task06 Lung

- **Use:** 3D CT segmentation of the Task06 foreground classes.
- **Official source:** [Task06 Lung archive](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar)
- **Dataset description and license:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- **License:** CC-BY-SA 4.0 as stated by the MSD data site.
- **Evaluation split:** deterministic held-out subset of the supplied training
  cases. The official test labels are not available to this project.
- **Limit:** the baseline uses a configurable subset of 16 labeled cases and
  ten epochs; its per-class Dice is an internal validation metric, not a robust
  generalization estimate.
- **Latest result:** 12 training / 4 validation volumes; cancer foreground
  Dice **0.00024**. This is effectively no useful segmentation on this split
  and should be presented as a failed baseline, not a performance claim.

The archive and extracted volumes are local DVC data and are not committed to
Git. Do not send the datasets to a public DVC remote.

`ml/import_msd_demo_series.py` can convert one local NIfTI training image to
freshly identified DICOM slices for a local Orthanc viewer demo. This is a
format-conversion helper, not a reconstruction of the source DICOM metadata.

## Data versioning

`dvc.yaml` defines classifier and segmentation training stages. The DVC cache
is local until a private remote is configured. A Git commit containing DVC
metadata alone does not make the underlying data available to collaborators.
