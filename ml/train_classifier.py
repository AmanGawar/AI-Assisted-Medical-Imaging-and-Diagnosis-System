import argparse
import io
import json
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pydicom
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

if __package__:
    from .rsna_data import RSNARecord, load_rsna_records
else:
    from rsna_data import RSNARecord, load_rsna_records

LABELS = ["Lung Opacity"]


class RSNAArchiveDataset(Dataset):
    def __init__(self, archive_path: Path, records: list[RSNARecord], augment: bool) -> None:
        self.archive = ZipFile(archive_path)
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                *([transforms.RandomHorizontalFlip()] if augment else []),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        dataset = pydicom.dcmread(io.BytesIO(self.archive.read(record.member)))
        pixels = dataset.pixel_array.astype(np.float32)
        pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0))
        pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
        low, high = np.percentile(pixels, [1, 99])
        if high <= low:
            high = low + 1.0
        pixels = np.clip((pixels - low) / (high - low), 0, 1)
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 1 - pixels
        image = Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(float(record.label), dtype=torch.float32),
        )

    def close(self) -> None:
        self.archive.close()


def select_records(records: list[RSNARecord], limit: int, seed: int) -> list[RSNARecord]:
    if limit < 0:
        raise ValueError("--max-samples must not be negative")
    if limit == 0 or limit >= len(records):
        return records
    if limit < 10:
        raise ValueError("--max-samples must be 0 (all) or at least 10")
    selected, _ = train_test_split(
        records,
        train_size=limit,
        random_state=seed,
        stratify=[record.label for record in records],
    )
    return list(selected)


def evaluate(model, loader, criterion, device) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    batches = 0
    targets: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            total_loss += float(criterion(logits, labels).item())
            batches += 1
            targets.extend(labels.int().cpu().tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    predictions = [int(score >= 0.5) for score in scores]
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        targets, predictions, labels=[0, 1]
    ).ravel()
    return {
        "bce": total_loss / max(batches, 1),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "accuracy_at_0_5": float(accuracy_score(targets, predictions)),
        "precision_at_0_5": float(precision_score(targets, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(targets, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(targets, predictions, zero_division=0)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "validation_samples": len(targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RSNA DenseNet121 baseline")
    parser.add_argument("--archive", type=Path, required=True, help="RSNA images ZIP")
    parser.add_argument("--annotations", type=Path, required=True, help="RSNA annotations JSON")
    parser.add_argument("--output", type=Path, default=Path("models/classifier.pt"))
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/classifier_metrics.json"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Stratified subset size; pass 0 to use every annotated study",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    records = select_records(
        load_rsna_records(args.archive, args.annotations), args.max_samples, args.seed
    )
    if len(records) < 10:
        raise ValueError("At least 10 labeled RSNA studies are required for a train/validation split")
    train_records, valid_records = train_test_split(
        records,
        test_size=0.2,
        random_state=args.seed,
        stratify=[record.label for record in records],
    )
    print(
        f"studies={len(records)} train={len(train_records)} validation={len(valid_records)} "
        f"positive={sum(record.label for record in records)}"
    )

    train_data = RSNAArchiveDataset(args.archive, train_records, augment=True)
    valid_data = RSNAArchiveDataset(args.archive, valid_records, augment=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    weights = None if args.no_pretrained else DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    positive_count = sum(record.label for record in train_records)
    negative_count = len(train_records) - positive_count
    train_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    validation_criterion = nn.BCEWithLogitsLoss()

    try:
        final_metrics: dict[str, float | int] = {}
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            batches = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = train_criterion(model(images).squeeze(1), labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1
            final_metrics = evaluate(
                model, valid_loader, validation_criterion, device
            )
            print(
                f"epoch={epoch + 1} train_bce={total_loss / max(batches, 1):.4f} "
                f"validation_bce={final_metrics['bce']:.4f} "
                f"validation_auc={final_metrics['roc_auc']:.4f} "
                f"validation_accuracy@0.5={final_metrics['accuracy_at_0_5']:.4f} "
                f"validation_f1@0.5={final_metrics['f1_at_0_5']:.4f}"
            )
    finally:
        train_data.close()
        valid_data.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "labels": LABELS,
            "model_version": "rsna-lung-opacity-densenet121-baseline",
        },
        args.output,
    )
    report = {
        "dataset": "RSNA Pneumonia Detection Challenge, initial annotations",
        "seed": args.seed,
        "studies": len(records),
        "training_studies": len(train_records),
        "validation_studies": len(valid_records),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "threshold": 0.5,
        "metrics": final_metrics,
        "evaluation_note": (
            "Held-out subset from the same source dataset; scores are preliminary "
            "and do not estimate clinical performance."
        ),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved research checkpoint: {args.output}; metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
