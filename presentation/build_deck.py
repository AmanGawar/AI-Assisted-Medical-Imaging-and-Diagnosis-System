import json
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "presentation" / "medical-imaging-demo.pptx"
BACKGROUND = RGBColor(12, 20, 34)
PANEL = RGBColor(19, 30, 46)
TEXT = RGBColor(231, 237, 247)
MUTED = RGBColor(160, 176, 199)
ACCENT = RGBColor(107, 211, 192)
WARNING = RGBColor(243, 201, 142)


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    size: int = 20,
    color: RGBColor = TEXT,
    bold: bool = False,
) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color


def add_panel(slide, x: float, y: float, width: float, height: float, text: str, size: int = 17) -> None:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = PANEL
    shape.line.color.rgb = RGBColor(50, 70, 93)
    shape.text_frame.clear()
    shape.text_frame.word_wrap = True
    shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = shape.text_frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = PP_ALIGN.CENTER
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.color.rgb = TEXT


def add_bullets(slide, lines: list[str], x: float = 0.9, y: float = 1.65, width: float = 11.4) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(4.9))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.level = 0
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(22)
        paragraph.font.color.rgb = TEXT
        paragraph.space_after = Pt(15)


def add_slide(prs: Presentation, title: str, subtitle: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BACKGROUND
    add_text(slide, title, 0.65, 0.35, 12.0, 0.6, 28, TEXT, True)
    if subtitle:
        add_text(slide, subtitle, 0.7, 1.0, 11.8, 0.35, 12, MUTED)
    add_text(
        slide,
        "FOR EDUCATION ONLY · NOT FOR CLINICAL USE",
        0.7,
        7.08,
        11.9,
        0.22,
        9,
        WARNING,
        True,
    )
    return slide


def format_metric(value: Any, suffix: str = "") -> str:
    if value is None:
        return "Not yet measured"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def build_deck() -> None:
    classifier = load_report(ROOT / "reports" / "classifier_metrics.json")
    segmentation = load_report(ROOT / "reports" / "segmentation_metrics.json")
    classifier_metrics = classifier.get("metrics", {})
    segmentation_metrics = segmentation.get("validation_foreground_dice_by_class", {})

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = add_slide(prs, "AI-Assisted Medical Imaging Review", "Educational prototype · RSNA chest X-ray classification · MSD 3D segmentation")
    add_text(slide, "DICOM ingestion → model-assisted review → human decision log", 0.85, 2.45, 11.5, 0.7, 24, ACCENT, True)
    add_text(slide, "A learning demo, not a medical device or diagnostic service.", 0.9, 3.35, 11.2, 0.5, 20, WARNING)

    slide = add_slide(prs, "The problem and project goal")
    add_bullets(
        slide,
        [
            "Medical images are large and complex; a research model can help demonstrate image classification and segmentation workflows.",
            "The prototype connects DICOM storage, model outputs, visual explanations, and a human review/audit flow.",
            "The goal is to demonstrate engineering and evaluation—not to make a clinical diagnosis.",
        ],
    )

    slide = add_slide(prs, "System architecture", "Local services are bound to loopback; model weights and datasets stay outside Git.")
    boxes = [
        (0.7, "DICOM upload\nOrthanc"),
        (3.15, "FastAPI\nstable-study worker"),
        (6.0, "DenseNet + Grad-CAM\nMONAI 3D U-Net"),
        (9.0, "React review UI\nSQLite audit"),
    ]
    for x, label in boxes:
        add_panel(slide, x, 2.35, 2.25, 1.25, label)
    for x in (2.65, 5.4, 8.25):
        arrow = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(2.72), Inches(0.42), Inches(0.42)
        )
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = ACCENT
        arrow.line.color.rgb = ACCENT
    add_text(slide, "DVC-tracked RSNA / MSD inputs → training and held-out evaluation reports", 1.1, 4.35, 11.2, 0.45, 18, MUTED)

    slide = add_slide(prs, "Data and model design")
    add_bullets(
        slide,
        [
            "RSNA 2018: DenseNet121 predicts Lung Opacity vs No Lung Opacity from one X-ray image.",
            "MSD Task06 Lung: MONAI 3D U-Net learns foreground segmentation from labeled CT volumes.",
            "DICOM pixel data are rescaled; CT slices are sorted by physical position and resampled consistently.",
            "Patient datasets and checkpoints are excluded from Git; use DVC with an approved private remote.",
        ],
    )

    slide = add_slide(prs, "Classifier evaluation", "Same-source, held-out subset; fixed threshold 0.5.")
    rows = [
        ("Studies", f"{classifier.get('training_studies', '—')} train / {classifier.get('validation_studies', '—')} validation"),
        ("ROC-AUC", format_metric(classifier_metrics.get("roc_auc"))),
        ("Accuracy", format_metric(classifier_metrics.get("accuracy_at_0_5"))),
        ("Precision / recall / F1", " / ".join(format_metric(classifier_metrics.get(key)) for key in ("precision_at_0_5", "recall_at_0_5", "f1_at_0_5"))),
        ("Confusion matrix", f"TN {classifier_metrics.get('true_negative', '—')} · FP {classifier_metrics.get('false_positive', '—')} · FN {classifier_metrics.get('false_negative', '—')} · TP {classifier_metrics.get('true_positive', '—')}"),
    ]
    for index, (label, value) in enumerate(rows):
        y = 1.55 + index * 0.85
        add_text(slide, label, 0.95, y, 3.4, 0.45, 18, MUTED, True)
        add_text(slide, value, 4.15, y, 7.7, 0.45, 20, TEXT)
    add_text(slide, classifier.get("evaluation_note", "Run training to generate a measured report."), 0.95, 6.15, 11.3, 0.55, 12, WARNING)

    slide = add_slide(prs, "3D segmentation evaluation", "Internal validation only; the official MSD test labels are not available.")
    if segmentation_metrics:
        add_bullets(
            slide,
            [
                f"{label}: Dice {format_metric(score)}"
                for label, score in segmentation_metrics.items()
            ]
            + [
                f"Mean foreground Dice: {format_metric(segmentation.get('mean_validation_foreground_dice'))}",
                f"Cases: {segmentation.get('train_cases', '—')} train / {segmentation.get('validation_cases', '—')} validation · epochs: {segmentation.get('epochs', '—')}",
            ],
        )
    else:
        add_bullets(
            slide,
            [
                "The MONAI 3D U-Net training path and CT-series inference endpoint are implemented.",
                "A measured Dice report will appear here after the MSD data are downloaded and training completes.",
                "Do not replace missing evaluation with an assumed or fabricated score.",
            ],
        )

    slide = add_slide(prs, "Demo walkthrough")
    add_bullets(
        slide,
        [
            "Import a permitted, non-identifiable study into Orthanc.",
            "Wait for the stable-study worker or select the study and run classification manually.",
            "Review the label, uncalibrated score, and Grad-CAM visualization.",
            "For a CT series with at least eight slices, run 3D segmentation and inspect its central resampled slice.",
            "Enter a reviewer identifier and record an educational approve/reject decision.",
        ],
    )

    slide = add_slide(prs, "Limits, safety, and next steps")
    add_bullets(
        slide,
        [
            "Scores are uncalibrated; Grad-CAM is coarse and does not establish a finding.",
            "Small subsets and internal splits do not demonstrate generalization or clinical utility.",
            "The prototype has no authentication, regulatory controls, or validated clinical workflow.",
            "Complete team-specific reflections, confirm data rights, and configure a private DVC remote before sharing.",
        ],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    build_deck()
