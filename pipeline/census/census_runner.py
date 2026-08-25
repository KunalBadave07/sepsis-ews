# pipeline/census/census_runner.py
"""
Background loop: fills empty beds with new simulated patient admissions,
and frees beds when a patient's data stream finishes (their "stay" ends).
"""
import glob
import random
import threading
import time
from pathlib import Path

from pipeline.census.factory import get_default_manager
from data.simulator.replay import replay_file

RAW_DIR = Path("data/raw/training_setA")
ADMIT_CHECK_INTERVAL_SECONDS = 10
REPLAY_SPEED = 0.3


def run_patient_stream(cm, bed: int, path: Path):
    patient_id = path.stem
    cm.admit_patient(bed, patient_id, str(path))
    print(f"[CENSUS] Admitted {patient_id} to bed {bed}")
    try:
        replay_file(path, speed=REPLAY_SPEED)
    finally:
        cm.discharge_bed(bed)
        print(f"[CENSUS] Bed {bed} discharged (was {patient_id})")


def main():
    cm = get_default_manager(total_beds=20)
    cm.init_beds()

    all_files = sorted(glob.glob(str(RAW_DIR / "*.psv")))
    if not all_files:
        raise RuntimeError("No .psv files found — check data/raw/training/ exists.")

    used_files = set()
    print(f"[CENSUS] Starting with {cm.total_beds} beds, {len(all_files)} patient files available.")

    while True:
        for bed in cm.get_empty_beds():
            available = [f for f in all_files if f not in used_files]
            if not available:
                # ran out of unique patients — this is a demo constraint,
                # not a real hospital constraint; reuse the pool
                used_files.clear()
                available = all_files

            chosen = random.choice(available)
            used_files.add(chosen)

            thread = threading.Thread(
                target=run_patient_stream, args=(cm, bed, Path(chosen)), daemon=True
            )
            thread.start()

        time.sleep(ADMIT_CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()