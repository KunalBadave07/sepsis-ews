# Sprint 4 Walkthrough — Full Containerization, Dashboard & Demo Packaging
### Every file, every command, in order. This is the finish line.

Before starting: make sure Sprint 3 is fully closed (`pytest tests/` all green), and Docker Desktop is running.

---

## DAY 1-3: Full Docker Compose Stack

### Step 1 — Create a single shared Dockerfile
All your Python services (API, feature pipeline, drift monitor, dashboard) share the same dependencies. Rather than maintaining four near-identical Dockerfiles, build one image and run each service with a different command.

New File → `infra/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "--version"]
```

### Step 2 — Expand docker-compose.yml to the full stack
Open `infra/docker-compose.yml` (the one from Sprint 1 that currently only has `redpanda` and `redis`) and replace its full contents:

```yaml
version: "3.8"

services:
  redpanda:
    image: docker.redpanda.com/redpandadata/redpanda:latest
    command:
      - redpanda start
      - --smp 1
      - --overprovisioned
      - --node-id 0
      - --kafka-addr PLAINTEXT://0.0.0.0:9092
      - --advertise-kafka-addr PLAINTEXT://redpanda:9092
    ports:
      - "9092:9092"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  mlflow-server:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    command: mlflow server --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --default-artifact-root /app/mlruns --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    depends_on:
      - postgres

  feature-pipeline:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    command: python pipeline/features/transforms.py
    depends_on:
      - redpanda
      - redis

  inference-api:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    depends_on:
      - redis

  drift-monitor:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    command: python monitoring/retrain_trigger/run_swadt_live.py
    depends_on:
      - inference-api

  dashboard:
    build:
      context: ..
      dockerfile: infra/Dockerfile
    command: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    environment:
      API_BASE_URL: "http://inference-api:8000"
    ports:
      - "8501:8501"
    depends_on:
      - inference-api

volumes:
  pgdata:
```

**Read this note before running anything:** notice `--advertise-kafka-addr PLAINTEXT://redpanda:9092` changed from `localhost:9092` to `redpanda:9092`. Inside Docker's internal network, containers reach each other by service name, not `localhost`. This means your Sprint 1 simulator script (`data/simulator/replay.py`), which you still run directly on your host machine (not in a container), needs to keep using `localhost:9092` — Docker Compose automatically exposes that port to your host. Don't get these two confused; it's the single most common container-networking mistake.

### Step 3 — Bring the whole stack up
```
docker compose -f infra/docker-compose.yml up --build
```
This will take several minutes the first time (building the shared image, pulling Postgres/Redpanda/Redis). Watch the logs scroll — you're looking for each service to report it started without crash-looping. `Ctrl+C` stops everything; add `-d` to run detached in the background instead once you trust it.

**Troubleshooting:** if `inference-api` keeps restarting, check its logs specifically:
```
docker compose -f infra/docker-compose.yml logs inference-api
```
The most likely cause is a missing environment variable or a path that only existed on your host machine (like a relative path to `ml/registry/latest_model.pkl` — confirm that file is actually committed or generated inside the container, since `.gitignore` may have excluded it; you may need a `COPY` step or a training step inside the Dockerfile/entrypoint to regenerate it).

### Step 4 — Confirm it's actually working end to end
- `http://localhost:8000/v1/health` should return `{"status": "ok"}`
- `http://localhost:5000` should show the MLflow UI
- `http://localhost:8501` should show the (still mostly empty — that's Day 4-6) Streamlit dashboard shell

Commit:
```
git add .
git commit -m "Sprint 4 Day 1-3: full 8-service Docker Compose stack running"
git push
```

---

## DAY 4-6: Streamlit Dashboard

### Step 5 — Install Streamlit locally for fast iteration
```
pip install streamlit requests
```
(You'll test locally against your API running on the host before trusting it inside Docker — much faster feedback loop.)

### Step 6 — Build the dashboard
New File → `dashboard/app.py`:
```python
# dashboard/app.py
import os
import random
import requests
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Sepsis-EWS Dashboard", layout="wide")
st.title("Sepsis-EWS — Live Monitoring Dashboard")
st.caption("Portfolio prototype using simulated/public research data — "
           "not real patients. See PRODUCTION_NOTES.md for full scope.")


def get_token() -> str:
    resp = requests.post(f"{API_BASE}/v1/auth/token",
                          data={"username": "nurse_jane", "password": "demo123"})
    resp.raise_for_status()
    return resp.json()["access_token"]


def predict(patient_id: str, vitals: dict) -> dict:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    resp = requests.post(f"{API_BASE}/v1/predict",
                          params={"patient_id": patient_id},
                          json=vitals, headers=headers)
    resp.raise_for_status()
    return resp.json()


def random_vitals() -> dict:
    hr = random.uniform(60, 140)
    sbp = random.uniform(70, 140)
    return {
        "heart_rate": hr, "resp_rate": random.uniform(12, 30), "sbp": sbp,
        "map_bp": random.uniform(50, 100), "temp_c": random.uniform(36, 39.5),
        "spo2": random.uniform(85, 100), "hr_rolling_mean": hr,
        "hr_rolling_std": random.uniform(1, 10),
        "map_rolling_mean": random.uniform(50, 100),
        "map_rolling_std": random.uniform(1, 10),
        "shock_index": hr / sbp if sbp else 0.0,
    }


if "token" not in st.session_state:
    st.session_state.token = get_token()

if "results" not in st.session_state:
    st.session_state.results = {}

PATIENTS = [f"p{i:06d}" for i in range(1, 6)]

if st.button("🔄 Refresh patient stream (simulate new readings)"):
    st.session_state.results = {}

for pid in PATIENTS:
    if pid not in st.session_state.results:
        st.session_state.results[pid] = predict(pid, random_vitals())

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Risk Overview")
    rows = [{
        "Patient": pid,
        "Risk Tier": r["risk_tier"],
        "Probability": round(r["probability"], 3),
        "Latency (ms)": r["latency_ms"],
    } for pid, r in st.session_state.results.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    selected = st.selectbox("Select a patient to inspect", PATIENTS)

with col2:
    st.subheader(f"SHAP Explanation — {selected}")
    result = st.session_state.results[selected]
    feat_df = pd.DataFrame(result["top_features"]).set_index("feature")
    st.bar_chart(feat_df)
    st.caption(f"Risk tier: **{result['risk_tier']}** — "
               f"probability {result['probability']:.3f}")

st.divider()
st.subheader("System Health")
try:
    health = requests.get(f"{API_BASE}/v1/health", timeout=3).json()
    st.success(f"API status: {health['status']}")
except Exception as e:
    st.error(f"API unreachable: {e}")
```

### Step 7 — Run it locally first (against your host API, not Docker yet)
Two terminals:
```
uvicorn api.main:app --reload --port 8000
```
```
streamlit run dashboard/app.py
```
It should open a browser tab automatically at `http://localhost:8501`. Click "Refresh patient stream," select different patients, confirm the bar chart updates with each one's SHAP features.

**If you get a 401 error on load:** your demo user credentials in `dashboard/app.py` need to match exactly what's in `api/security/auth.py`'s `FAKE_USERS` dict from Sprint 3.

### Step 8 — Test it for real inside Docker
```
docker compose -f infra/docker-compose.yml up --build dashboard inference-api redis
```
Open `http://localhost:8501` again. This time the dashboard container is calling `inference-api:8000` internally (per the `API_BASE_URL` environment variable set in `docker-compose.yml`), not your host's `localhost`. If this fails but Step 7 worked, it's almost certainly that env var not being picked up — double check the `environment:` block under `dashboard` in your compose file.

Commit:
```
git add .
git commit -m "Sprint 4 Day 4-6: Streamlit dashboard working locally and in Docker"
git push
```

---

## DAY 7-8: GitHub Actions CI

### Step 9 — Expand your CI workflow
Open (or create) `.github/workflows/ci.yml`:
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  docker-build-check:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f infra/Dockerfile -t sepsis-ews:ci .
```
`needs: test` means the Docker build job only runs if your test suite passes first — no point building an image around code you already know is broken.

### Step 10 — Push and watch it run
```
git add .
git commit -m "Sprint 4 Day 7-8: CI runs full test suite + docker build check on every push"
git push
```
Go to your repo on GitHub, click the **Actions** tab, and watch both jobs run. Green checkmarks on both = your CI pipeline is real, not decorative.

**If the test job fails in CI but passed locally:** the most common cause is a missing file your tests depend on that's excluded by `.gitignore` (like `ml/registry/latest_model.pkl`) — CI runs on a completely fresh checkout, so anything your tests need must either be committed or regenerated as a CI step.

---

## DAY 9-10: README, Demo Recording & Final Packaging

### Step 11 — Write the README
New File → `README.md` in the project root:
```markdown
# Sepsis-EWS: Real-Time Sepsis Deterioration Early-Warning System

A production-grade streaming ML system demonstrating real-time clinical
risk scoring, explainable AI, and a novel adaptive drift-detection
mechanism (SWADT). Built as a portfolio prototype on public research
data — see [PRODUCTION_NOTES.md](./PRODUCTION_NOTES.md) for exactly
what's real, what's simulated, and what real deployment would require.

## Architecture

\`\`\`mermaid
flowchart LR
    Sim[Patient Simulator] --> RP[(Redpanda)]
    RP --> Val[Pydantic Validator]
    Val -->|clean| Feat[Polars Feature Engine]
    Val -->|invalid| DLQ[(Dead Letter Queue)]
    Feat --> Feast[(Feast / Redis)]
    Feast --> API[FastAPI + SHAP Explainability]
    API --> Dash[Streamlit Dashboard]
    API --> Audit[(Audit Log)]
    SWADT[SWADT Drift Monitor] --> API
\`\`\`

## Quickstart
\`\`\`
docker compose -f infra/docker-compose.yml up --build
\`\`\`
Then open:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

## Benchmark Results
| Metric | Value |
|---|---|
| PR-AUC | *(fill in your real Sprint 2 number)* |
| F-beta (β=2) | *(fill in)* |
| Brier score | *(fill in)* |
| SHAP explanation latency | *(fill in your measured number)* |

## Key Design Decisions
- **SWADT** (SHAP-Weighted Adaptive Drift-Threshold): a novel mechanism
  fusing per-feature distributional drift with live SHAP importance
  velocity to reduce false retraining triggers. See the full technical
  paper: `SWADT-research-paper.docx`.
- **Structurally advisory-only**: the API's response schema has no
  field for treatment/medication/dosage — enforced by a passing test,
  not just design intent.
- **Full audit trail**: every prediction is logged with a human
  acknowledgment step, answering the liability/guardrails question
  directly rather than assuming it away.

## Repository Structure
See `infra/docker-compose.yml` for the full service topology.
```
Fill in the benchmark table with your *actual* numbers from Sprint 2 — don't leave placeholder text in a real README, that's worse than an honest modest number.

### Step 12 — Record the demo
- **Windows:** `Win + G` opens the Xbox Game Bar recorder.
- **Mac:** `Cmd + Shift + 5`.
- Keep it under 90 seconds: run `docker compose up`, show the dashboard populating, click one patient's SHAP chart, done.
- Save it into the repo (e.g. `docs/demo.gif` if you convert it, or link a short unlisted YouTube/Loom upload in the README) — a README with an embedded visual gets dramatically more attention than a wall of text.

### Step 13 — Final tag and push
```
git add .
git commit -m "Sprint 4 Day 9-10: README, architecture diagram, demo recording — v1.0"
git tag v1.0
git push origin main --tags
```

---

## Sprint 4 — Definition of Done
- [ ] `docker compose -f infra/docker-compose.yml up --build` brings up all 8 services cleanly from a fresh clone
- [ ] Dashboard shows live patient risk tiers and a working SHAP bar chart, running inside Docker (not just locally)
- [ ] GitHub Actions shows green on both the test job and the Docker build job
- [ ] README has a real (filled-in, not placeholder) benchmark table, a Mermaid diagram, and links `PRODUCTION_NOTES.md` prominently
- [ ] A demo recording exists and is linked or embedded

When all five are checked, the whole project — Sprints 1 through 4 — is genuinely complete. That's the point where the Spark/Databricks side-quest we talked about slots in cleanly: rebuilding `ml/training/build_dataset.py` as a PySpark job, separately, honestly labeled as an additional skills demonstration rather than a core dependency of the live system.
