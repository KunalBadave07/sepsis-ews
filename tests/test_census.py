# tests/test_census.py
import fakeredis
from pipeline.census.census_manager import CensusManager


def test_admit_and_discharge_cycle():
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    cm = CensusManager(client, total_beds=3)
    cm.init_beds()

    assert set(cm.get_empty_beds()) == {1, 2, 3}

    cm.admit_patient(1, "p000001", "data/raw/training/p000001.psv")
    assert set(cm.get_empty_beds()) == {2, 3}

    census = cm.get_census()
    occupied = [b for b in census if b["status"] == "occupied"]
    assert len(occupied) == 1
    assert occupied[0]["patient_id"] == "p000001"

    cm.discharge_bed(1)
    assert set(cm.get_empty_beds()) == {1, 2, 3}