# monitoring/audit/audit_log.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("monitoring/audit/audit.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            patient_id TEXT,
            probability REAL,
            risk_tier TEXT,
            top_features TEXT,
            model_version TEXT,
            latency_ms REAL,
            requested_by TEXT,
            created_at TEXT,
            disposition TEXT,
            clinician_note TEXT,
            acknowledged_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(prediction_id, patient_id, probability, risk_tier,
                    top_features, model_version, latency_ms, requested_by):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (prediction_id, patient_id, probability, risk_tier, top_features,
            model_version, latency_ms, requested_by, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (prediction_id, patient_id, probability, risk_tier,
         json.dumps(top_features), model_version, latency_ms,
         requested_by, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def log_acknowledgment(prediction_id, disposition, clinician_note):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """UPDATE predictions SET disposition = ?, clinician_note = ?,
           acknowledged_at = ? WHERE prediction_id = ?""",
        (disposition, clinician_note, datetime.utcnow().isoformat(), prediction_id),
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()
    return updated > 0


def get_prediction(prediction_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE prediction_id = ?",
                        (prediction_id,)).fetchone()
    conn.close()
    return dict(row) if row else None