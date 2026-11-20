import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.segmentation import build_unet


def load_cases(data_dir: Path, maximum: int, seed: int) -> tuple[list[dict[str, str]], dict[int, str]]:
    metadata = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
    raw_labels = metadata.get("labels")
    if not isinstance(raw_labels, dict):
        raise ValueError("dataset.json must define a label map")
    labels = {int(key): str(value) for key, value in raw_labels.items()}
    if sorted(labels) != list(range(len(labels))) or len(labels) < 2:
        raise ValueError("dataset.json labels must be contiguous and include background")

    data_root = data_dir.resolve()
    cases = []
    for item in metadata.get("training", []):
        case: dict[str, str] = {}
        for key in ("image", "label"):
            candidate = (data_root / str(item[key]).removeprefix("./")).resolve()
            try:
                candidate.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(f"dataset.json {key} path escapes the dataset directory") from exc
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            case[key] = str(candidate)
        cases.append(case)
    if len(cases) < 2:
        raise ValueError("At least two labeled volumes are required for train/validation evaluation")
    if maximum < 0:
        raise ValueError("--max-cases must not be negative")
    if maximum and len(cases) > maximum:
        cases = random.Random(seed).sample(cases, maximum)
    return cases, labels


def foreground_dice(
    prediction: torch.Tensor, target: torch.Tensor, labels: dict[int, str]
) -> dict[str, float | None]:
    scores: dict[str, list[float]] = {name: [] for index, name in labels.items() if index > 0}
    for batch_index in range(prediction.shape[0]):
        predicted_volume = prediction[batch_index]
        target_volume = target[batch_index, 0]
        for label, name in labels.items():
            if label == 0:
                continue
            predicted_mask = predicted_volume == label
            target_mask = target_volume == label
            denominator = int(predicted_mask.sum().item() + target_mask.sum().item())
            if denominator:
                intersection = int((predicted_mask & target_mask).sum().item())
                scores[name].append(2.0 * intersection / denominator)
    return {
        name: float(np.mean(values)) if values else None
        for name, values in scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an MSD MONAI 3D U-Net baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/segmentation.pt"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/segmentation_metrics.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0, help="Random subset; 0 uses all cases")
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.roi_size < 16:
        raise ValueError("epochs and batch-size must be positive; roi-size must be at least 16")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")

    set_determinism(seed=args.seed)
    cases, labels = load_cases(args.data_dir, args.max_cases, args.seed)
    shuffled = random.Random(args.seed).sample(cases, len(cases))
    validation_count = max(1, int(round(len(shuffled) * args.validation_fraction)))
    validation_cases = shuffled[:validation_count]
    training_cases = shuffled[validation_count:]
    if not training_cases:
        raise ValueError("The training split is empty; provide at least two cases")
    roi_size = (args.roi_size,) * 3
    keys = ("image", "label")
    shared = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True
        ),
    ]
    train_transforms = Compose(
        [
            *shared,
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            EnsureTyped(keys=keys),
        ]
    )
    validation_transforms = Compose([*shared, EnsureTyped(keys=keys)])
    training_loader = DataLoader(
        Dataset(data=training_cases, transform=train_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Dataset(data=validation_cases, transform=validation_transforms),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(len(labels)).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(
        f"device={device} train_cases={len(training_cases)} validation_cases={len(validation_cases)} "
        f"labels={labels}"
    )

    validation_metrics: dict[str, float | None] = {}
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in training_loader:
            images = batch["image"].to(device)
            targets = batch["label"].long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        model.eval()
        epoch_metrics: dict[str, list[float]] = {
            name: [] for index, name in labels.items() if index > 0
        }
        with torch.inference_mode():
            for batch in validation_loader:
                images = batch["image"].to(device)
                targets = batch["label"].long().to(device)
                logits = sliding_window_inference(
                    images,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                )
                epoch_dice = foreground_dice(logits.argmax(dim=1), targets, labels)
                for name, value in epoch_dice.items():
                    if value is not None:
                        epoch_metrics[name].append(value)
        validation_metrics = {
            name: float(np.mean(values)) if values else None
            for name, values in epoch_metrics.items()
        }
        available_scores = [value for value in validation_metrics.values() if value is not None]
        mean_dice = float(np.mean(available_scores)) if available_scores else None
        print(
            f"epoch={epoch + 1} train_dice_ce={total_loss / max(batch_count, 1):.4f} "
            f"validation_foreground_dice={mean_dice}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.cpu().state_dict(),
        "labels": labels,
        "model_version": "msd-task06-monai-3d-unet-baseline",
        "roi_size": roi_size,
    }
    torch.save(checkpoint, args.output)

    report = {
        "dataset": "Medical Segmentation Decathlon Task06_Lung",
        "seed": args.seed,
        "train_cases": len(training_cases),
        "validation_cases": len(validation_cases),
        "epochs": args.epochs,
        "roi_size": roi_size,
        "validation_foreground_dice_by_class": validation_metrics,
        "mean_validation_foreground_dice": (
            float(np.mean([value for value in validation_metrics.values() if value is not None]))
            if any(value is not None for value in validation_metrics.values())
            else None
        ),
        "evaluation_note": "Internal held-out subset only; not a clinical performance estimate.",
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved_checkpoint={args.output} metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
