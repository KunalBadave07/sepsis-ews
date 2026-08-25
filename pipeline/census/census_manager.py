# pipeline/census/census_manager.py
"""
Tracks which beds are occupied by which simulated patients, backed by
Redis. Built as an injectable class (not module-level global state) so
it can be unit-tested against fakeredis without a real Redis server.
"""
import json
from datetime import datetime


class CensusManager:
    def __init__(self, redis_client, total_beds: int = 20):
        self.r = redis_client
        self.total_beds = total_beds

    def init_beds(self):
        for bed in range(1, self.total_beds + 1):
            key = f"census:bed:{bed}"
            if not self.r.exists(key):
                self.r.set(key, json.dumps({"status": "empty"}))

    def get_empty_beds(self) -> list[int]:
        empty = []
        for bed in range(1, self.total_beds + 1):
            data = self._get_bed(bed)
            if data.get("status") == "empty":
                empty.append(bed)
        return empty

    def admit_patient(self, bed: int, patient_id: str, source_file: str):
        self.r.set(f"census:bed:{bed}", json.dumps({
            "status": "occupied",
            "patient_id": patient_id,
            "admitted_at": datetime.utcnow().isoformat(),
            "source_file": source_file,
        }))

    def discharge_bed(self, bed: int):
        self.r.set(f"census:bed:{bed}", json.dumps({"status": "empty"}))

    def get_census(self) -> list[dict]:
        return [{"bed": bed, **self._get_bed(bed)} for bed in range(1, self.total_beds + 1)]

    def _get_bed(self, bed: int) -> dict:
        raw = self.r.get(f"census:bed:{bed}")
        return json.loads(raw) if raw else {"status": "empty"}