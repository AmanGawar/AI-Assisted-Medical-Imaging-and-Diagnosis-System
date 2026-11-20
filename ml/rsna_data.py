import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


@dataclass(frozen=True)
class RSNARecord:
    study_uid: str
    member: str
    label: int


def load_rsna_records(archive_path: Path, annotations_path: Path) -> list[RSNARecord]:
    with annotations_path.open(encoding="utf-8") as file:
        project = json.load(file)

    label_names = {
        label["id"]: label["name"]
        for group in project.get("labelGroups", [])
        for label in group.get("labels", [])
    }
    positive_ids = {label_id for label_id, name in label_names.items() if name == "Lung Opacity"}
    negative_ids = {
        label_id for label_id, name in label_names.items() if name == "No Lung Opacity"
    }
    if not positive_ids or not negative_ids:
        raise ValueError("RSNA annotation file must define Lung Opacity and No Lung Opacity")

    datasets = project.get("datasets", [])
    if len(datasets) != 1:
        raise ValueError("Expected exactly one RSNA dataset in the annotation file")
    dataset = datasets[0]

    study_annotations: dict[str, set[str]] = {}
    study_sops: dict[str, set[str]] = {}
    for annotation in dataset.get("annotations", []):
        study_uid = annotation.get("StudyInstanceUID")
        sop_uid = annotation.get("SOPInstanceUID")
        label_id = annotation.get("labelId")
        if not study_uid or not sop_uid:
            raise ValueError("RSNA annotation is missing a DICOM UID")
        if label_id in positive_ids:
            label = "positive"
        elif label_id in negative_ids:
            label = "negative"
        else:
            raise ValueError(f"Unexpected RSNA annotation label: {label_id!r}")
        study_annotations.setdefault(study_uid, set()).add(label)
        study_sops.setdefault(study_uid, set()).add(sop_uid)

    with ZipFile(archive_path) as archive:
        members_by_sop: dict[str, str] = {}
        for member in archive.namelist():
            path = PurePosixPath(member)
            if path.suffix.lower() != ".dcm":
                continue
            sop_uid = path.stem
            if sop_uid in members_by_sop:
                raise ValueError(f"Duplicate DICOM SOPInstanceUID in archive: {sop_uid}")
            members_by_sop[sop_uid] = member

    records: list[RSNARecord] = []
    seen_studies: set[str] = set()
    for study in dataset.get("studies", []):
        study_uid = study.get("StudyInstanceUID")
        if not study_uid or study_uid in seen_studies:
            raise ValueError("RSNA study list contains a missing or duplicate StudyInstanceUID")
        seen_studies.add(study_uid)

        labels = study_annotations.get(study_uid, set())
        if len(labels) != 1:
            raise ValueError(
                f"Study {study_uid} must have exactly one consistent binary label"
            )
        sop_uids = study_sops[study_uid]
        if len(sop_uids) != 1:
            raise ValueError(f"Study {study_uid} does not map to exactly one DICOM instance")
        sop_uid = next(iter(sop_uids))
        member = members_by_sop.get(sop_uid)
        if member is None:
            raise ValueError(f"No DICOM file in archive for study {study_uid}")
        records.append(RSNARecord(study_uid, member, int("positive" in labels)))

    if len(records) != len(study_annotations):
        raise ValueError("RSNA annotations reference studies missing from the study list")
    return records
