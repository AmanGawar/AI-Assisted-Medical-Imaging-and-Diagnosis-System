import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


@contextmanager
def connect(database: str) -> Iterator[sqlite3.Connection]:
    path = Path(database)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize(database: str) -> None:
    with connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                study_id TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                decision TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            );
            CREATE TABLE IF NOT EXISTS segmentations (
                id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS study_automation (
                study_id TEXT PRIMARY KEY,
                instance_count INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('processing', 'completed', 'failed')),
                analysis_id TEXT,
                segmentation_id TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        automation_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(study_automation)")
        }
        if "segmentation_id" not in automation_columns:
            connection.execute(
                "ALTER TABLE study_automation ADD COLUMN segmentation_id TEXT"
            )


def save_analysis(database: str, study_id: str, result: dict[str, Any]) -> str:
    analysis_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        connection.execute(
            "INSERT INTO analyses (id, study_id, result_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (analysis_id, study_id, json.dumps(result), created_at),
        )
    return analysis_id


def get_latest_analysis(database: str, study_id: str) -> dict[str, Any] | None:
    with connect(database) as connection:
        row = connection.execute(
            "SELECT id, result_json, created_at FROM analyses "
            "WHERE study_id = ? ORDER BY created_at DESC LIMIT 1",
            (study_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        **json.loads(row["result_json"]),
        "analysis_id": row["id"],
        "created_at": row["created_at"],
    }


def save_review(
    database: str,
    analysis_id: str,
    study_id: str,
    reviewer: str,
    decision: str,
    note: str,
) -> dict[str, str] | None:
    review_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        analysis = connection.execute(
            "SELECT study_id FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        if analysis is None or analysis["study_id"] != study_id:
            return None
        connection.execute(
            "INSERT INTO reviews "
            "(id, analysis_id, study_id, reviewer, decision, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (review_id, analysis_id, study_id, reviewer, decision, note, created_at),
        )
    return {
        "id": review_id,
        "analysis_id": analysis_id,
        "study_id": study_id,
        "reviewer": reviewer,
        "decision": decision,
        "note": note,
        "created_at": created_at,
    }


def get_reviews(database: str, study_id: str) -> list[dict[str, str]]:
    with connect(database) as connection:
        rows = connection.execute(
            "SELECT id, analysis_id, study_id, reviewer, decision, note, created_at "
            "FROM reviews WHERE study_id = ? ORDER BY created_at",
            (study_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_segmentation(
    database: str, study_id: str, result: dict[str, Any]
) -> str:
    segmentation_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        connection.execute(
            "INSERT INTO segmentations (id, study_id, result_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (segmentation_id, study_id, json.dumps(result), created_at),
        )
    return segmentation_id


def get_segmentations(database: str, study_id: str) -> list[dict[str, Any]]:
    with connect(database) as connection:
        rows = connection.execute(
            "SELECT id, study_id, result_json, created_at "
            "FROM segmentations WHERE study_id = ? ORDER BY created_at",
            (study_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "study_id": row["study_id"],
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def has_analysis(database: str, study_id: str) -> bool:
    with connect(database) as connection:
        row = connection.execute(
            "SELECT 1 FROM analyses WHERE study_id = ? LIMIT 1", (study_id,)
        ).fetchone()
    return row is not None


def has_segmentation(database: str, study_id: str) -> bool:
    with connect(database) as connection:
        row = connection.execute(
            "SELECT 1 FROM segmentations WHERE study_id = ? LIMIT 1", (study_id,)
        ).fetchone()
    return row is not None


def get_automation_status(database: str, study_id: str) -> dict[str, Any] | None:
    with connect(database) as connection:
        row = connection.execute(
            "SELECT study_id, instance_count, status, analysis_id, segmentation_id, "
            "error, updated_at "
            "FROM study_automation WHERE study_id = ?",
            (study_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def list_automation_status(database: str) -> list[dict[str, Any]]:
    with connect(database) as connection:
        rows = connection.execute(
            "SELECT study_id, instance_count, status, analysis_id, segmentation_id, "
            "error, updated_at "
            "FROM study_automation ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def save_automation_status(
    database: str,
    study_id: str,
    instance_count: int,
    status: str,
    analysis_id: str | None = None,
    segmentation_id: str | None = None,
    error: str | None = None,
) -> None:
    if status not in {"processing", "completed", "failed"}:
        raise ValueError(f"Unsupported study automation status: {status}")
    updated_at = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        connection.execute(
            "INSERT INTO study_automation "
            "(study_id, instance_count, status, analysis_id, segmentation_id, error, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(study_id) DO UPDATE SET "
            "instance_count=excluded.instance_count, status=excluded.status, "
            "analysis_id=excluded.analysis_id, segmentation_id=excluded.segmentation_id, "
            "error=excluded.error, "
            "updated_at=excluded.updated_at",
            (
                study_id,
                instance_count,
                status,
                analysis_id,
                segmentation_id,
                error,
                updated_at,
            ),
        )
