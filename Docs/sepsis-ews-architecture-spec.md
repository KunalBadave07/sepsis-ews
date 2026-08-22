# Real-Time Multimodal Sepsis Deterioration Early-Warning System (Sepsis-EWS)
### End-to-End Production ML Architecture Specification

---

## Why This Beats 99% of Portfolio Projects

Most "ML classification" portfolio projects are static Kaggle-CSV-to-accuracy-score exercises. That's a homework assignment, not engineering. This spec is deliberately hard in the ways that matter to a hiring manager:

1. **Extreme class imbalance** (sepsis onset is ~2-8% of ICU stays) — you can't cheat with accuracy, you have to justify PR-AUC and cost-weighted metrics.
2. **Non-stationary data** — patient physiology drifts hour to hour; a model trained on admission data silently rots by hour 12. This forces you to build drift detection, not just a model.
3. **Real-time constraint** — sub-second scoring on streaming vitals is an entirely different engineering problem than batch `model.predict(df)`.
4. **Explainability is a hard requirement, not a nice-to-have** — in healthcare, a black-box "87% probability of sepsis" is legally and clinically useless. You need per-prediction SHAP attribution a clinician can act on in under 50ms.
5. **It maps directly to three job families** (ML Engineer, Data Engineer, Backend Engineer) because it forces you to touch feature stores, streaming pipelines, model serving, and monitoring — the actual surface area of a production ML system.

**The problem statement:** Given a continuous stream of ICU vitals, labs, and clinical event logs, classify each patient-hour into {stable, at-risk, deteriorating} for sepsis onset, with a decision window that gives clinicians actionable lead time (target: 4-6 hours before Sepsis-3 criteria are met), while adapting online to distributional drift across patient cohorts and ICU sites.

### Benchmark Datasets (real, public, industry-standard)

| Dataset | Use | Schema Notes |
|---|---|---|
| **MIMIC-IV** (PhysioNet, requires credentialed access via CITI training) | Primary — ICU vitals, labs, clinical notes, timestamped events | `chartevents`, `labevents`, `icustays` tables; join on `subject_id`, `hadm_id`, `stay_id` |
| **PhysioNet/Computing in Cardiology Challenge 2019 (Sepsis)** | Secondary — pre-labeled hourly sepsis onset, no credentialing needed, great for fast bootstrapping | 40+ hourly features (HR, O2Sat, Temp, SBP/MAP/DBP, Resp, labs), binary `SepsisLabel` per hour |
| **eICU Collaborative Research Database** | Multi-site validation — tests cross-hospital drift (different equipment/protocols = real distribution shift) | Similar schema to MIMIC, different hospital IDs give you a natural drift testbed |

Use the **2019 PhysioNet Sepsis Challenge dataset** as your fast-iteration dev set (no credentialing wait, already hourly-windowed) and **MIMIC-IV** as the "production realism" dataset for the portfolio narrative — this combo lets you start building immediately while still name-dropping the harder dataset.

---

## Module 2: Data Engineering & Streaming Architecture

### Pipeline Topology

```
[Patient Monitor Simulator] → [Redpanda topic: vitals.raw]
                                     │
                        [Pydantic validation consumer]
                                     │
                          ┌──────────┴──────────┐
                    [reject → DLQ topic]   [valid → vitals.clean]
                                                  │
                                    [Polars streaming feature transform]
                                                  │
                                    [Feast online store (Redis)]
                                                  │
                                    [FastAPI inference service]
```

**Ingestion:** Simulate the ICU monitor stream with a Python generator that replays MIMIC/PhysioNet rows at accelerated wall-clock time onto a Redpanda topic (`vitals.raw`) — Redpanda over Kafka because it's a single binary, no ZooKeeper, and trivially runs in Docker Compose on a laptop while being Kafka-API-compatible (so the resume line "built a Kafka-compatible streaming pipeline" is 100% true).

**Validation contract (Pydantic v2):**
```python
class VitalReading(BaseModel):
    patient_id: str
    timestamp: datetime
    heart_rate: float = Field(ge=0, le=300)
    resp_rate: float = Field(ge=0, le=80)
    sbp: float = Field(ge=0, le=300)
    map_bp: float = Field(ge=0, le=250)
    temp_c: float = Field(ge=25, le=45)
    spo2: float = Field(ge=0, le=100)
    wbc: float | None = Field(default=None, ge=0, le=100)
    lactate: float | None = Field(default=None, ge=0, le=30)

    @field_validator("timestamp")
    @classmethod
    def not_future(cls, v):
        if v > datetime.utcnow():
            raise ValueError("future timestamp — clock skew or bad sim data")
        return v
```
Rows that fail validation route to a `vitals.dlq` topic with the exception payload attached — this is the pattern that separates a real pipeline from a script (Great Expectations is heavier and better suited to *batch* data quality checks on training data; use it for the offline training pipeline, Pydantic for hot-path streaming validation).

**Feature engineering (Polars, not Pandas, for the streaming hot path):**
- Multi-resolution rolling windows: 1h, 4h, 8h rolling mean/std/slope for HR, RR, MAP, Temp, SpO2
- Rolling quantile deviation score: `(current_value - rolling_median) / rolling_IQR` — robust to outliers common in noisy telemetry
- Shock index (`HR / SBP`) and modified shock index — clinically validated composite features that boost signal
- Time-since-last-abnormal-lab (censored duration feature)
- SIRS/qSOFA criteria count as engineered binary features (domain knowledge baked into features, always a strong signal for reviewers that you understand the problem domain, not just the tooling)

**Feature Store:** Feast with Redis as the online store. Offline store = Parquet on disk (or S3 in cloud-ready config). This gives you point-in-time-correct joins for training (avoiding label leakage — a classic mistake in time-series clinical ML where future data leaks into features) and sub-10ms feature retrieval at inference.

---

## Module 3: ML & Adaptive Classification Engine

### Handling Extreme Imbalance
- **Primary approach:** Cost-sensitive learning via `scale_pos_weight` in XGBoost tuned via Bayesian optimization (Optuna) against **F-beta (β=2)** rather than F1 — false negatives (missed sepsis) are far more costly than false positives (extra clinician review).
- **Secondary comparison:** Focal loss custom objective (`γ=2, α=0.75`) implemented as an XGBoost custom objective function — include this even though cost-sensitive weighting usually wins, because benchmarking both and *explaining why one won* is what separates an engineer from someone who just calls `.fit()`.
- **Dynamic threshold optimization:** Don't ship a fixed 0.5 threshold. Run a continuous threshold-tuning job that optimizes the decision boundary against a cost matrix (cost of missed sepsis vs. cost of false alarm/alert fatigue) recomputed weekly on a rolling validation window.

### Model Architecture
- **Offline champion model:** LightGBM (faster to retrain than XGBoost at this feature count, comparable accuracy) trained on the full historical batch, versioned and registered.
- **Online adaptive layer:** River's `ADWIN`-wrapped `HoeffdingTreeClassifier` or `AdaptiveRandomForestClassifier` running in shadow mode alongside the batch model — when ADWIN detects a change point (concept drift) in the online model's error rate, it fires a retraining trigger for the offline model rather than fully replacing it (streaming models alone are usually weaker than a well-tuned gradient booster; the online layer's real job here is *drift sensing*, not primary prediction — say this explicitly in your README, it shows engineering judgment).

### Evaluation Matrix
| Metric | Why |
|---|---|
| PR-AUC | Correct primary metric under imbalance — ROC-AUC is misleadingly optimistic here |
| F-beta (β=2) | Weights recall higher — reflects clinical cost asymmetry |
| Cost-weighted accuracy | Custom cost matrix: FN=100, FP=5, TN=0, TP=0 (tune with clinical literature-backed estimates) |
| Brier score | Calibration — a 0.9 "probability" should mean 90% actually deteriorate, critical for clinician trust |
| Lead-time distribution | Not a classic ML metric, but *the* metric that matters clinically — histogram of hours-before-onset the model fired |

### Explainability Pipeline
- Pre-compute SHAP `TreeExplainer` background dataset (K-means summarized, ~100 representative samples) offline — computing full SHAP on the fly is too slow.
- At inference, compute per-prediction SHAP values against the cached background — TreeExplainer on a summarized background gets you comfortably under 50ms for a single row.
- Serve top-5 SHAP feature attributions alongside every prediction in the API response, structured for direct rendering as a waterfall chart in the dashboard.
- Bonus point that will actually get noticed: implement one **counterfactual explanation** endpoint ("what change in MAP/lactate would flip this patient from at-risk to stable") using DiCE — this is rare in portfolio projects and directly demonstrates you understand the difference between explaining a prediction and explaining an *actionable* prediction.

---

## Module 4: Streaming MLOps, Drift Detection & Retraining

### Drift Monitoring
- **Evidently AI** for scheduled batch drift reports (data drift via PSI/KS-test per feature, target drift, prediction drift) — run hourly against a rolling reference window.
- **Alibi-Detect** for the streaming path — an online `KSDrift` or `MMDDrift` detector consuming the same feature stream in near-real-time, cheaper to run continuously than full Evidently reports.

### Trigger Matrix

| Signal | Threshold | Action |
|---|---|---|
| PSI per feature | > 0.25 | Flag feature, log to monitoring dashboard |
| KS-test p-value | < 0.01 on 3 consecutive windows | Trigger retraining pipeline |
| Prediction distribution shift | Alibi-Detect MMD p-value < 0.05 | Page on-call + trigger retrain |
| ADWIN change point (online model) | Detected | Trigger retrain + shorten next monitoring window |
| Model calibration decay | Brier score degrades >15% vs. baseline | Trigger recalibration (Platt scaling refresh) before full retrain |

### Experiment Tracking & Registry
- **MLflow** for run tracking (hyperparams, metrics, SHAP summary artifacts) and Model Registry (Staging → Production → Archived lifecycle, with the retraining pipeline auto-promoting a challenger only if it beats the champion on the full eval matrix, not just accuracy).
- **DVC** for dataset and feature-store snapshot versioning, so every registered model has a reproducible pointer back to the exact training data version — this is the detail that makes a reviewer believe you've actually worked in a regulated-data environment.

---

## Module 5: Enterprise Software Architecture & API Specs

### Backend
FastAPI (async), served via Uvicorn/Gunicorn workers. Key endpoints:
- `POST /v1/predict` — single patient-hour scoring, returns `{probability, risk_tier, shap_top_features, model_version}`
- `POST /v1/predict/counterfactual` — DiCE-based counterfactual
- `GET /v1/monitor/drift` — current drift status per feature
- `GET /v1/health` — liveness/readiness, includes model load status and feature-store connectivity check
- WebSocket `/v1/stream` — pushes live scored events to the dashboard as they're processed

Redis doubles as both the Feast online store and the API response cache for repeated feature lookups within the same patient-hour.

### Containerization
```
docker-compose.yml
├── redpanda (streaming broker)
├── redis (feature store online + cache)
├── postgres (MLflow backend store + Feast registry)
├── mlflow-server
├── feature-pipeline (Polars consumer)
├── inference-api (FastAPI)
├── drift-monitor (scheduled Evidently/Alibi-Detect worker)
└── dashboard (Streamlit or Next.js)
```
Single `make run` / `docker compose up` brings up the full stack locally — no cloud dependency required to demo it end to end, but every service is written to swap to managed equivalents (MSK/Confluent Cloud, ElastiCache, RDS) via env vars only.

### Dashboard Spec
- Live scrolling patient stream table with color-coded risk tier
- Real-time confidence gauge with the current alert threshold overlaid
- Click-through SHAP waterfall per patient-hour
- System panel: p50/p95/p99 inference latency, throughput, drift radar chart (per-feature PSI as a radar/spider plot — visually distinctive and instantly communicates "this person thought about ops")

---

## Module 6: Sprint Plan & Repo Structure

### 4-Phase Roadmap
- **Phase 1 (Week 1-2):** Redpanda ingestion + Pydantic validation + Polars feature pipeline + Feast online store wired end to end. Deliverable: raw vitals in → validated features in Redis out.
- **Phase 2 (Week 3-4):** Offline training pipeline (LightGBM + Optuna), full eval matrix, SHAP explainability pipeline, MLflow tracking. Deliverable: registered model with reproducible metrics.
- **Phase 3 (Week 5-6):** River online layer + ADWIN drift sensing + Evidently/Alibi-Detect batch+streaming drift monitors + automated retraining trigger + FastAPI serving layer. Deliverable: closed-loop drift → retrain → redeploy.
- **Phase 4 (Week 7-8):** Full Docker Compose stack, Streamlit/Next.js dashboard, CI/CD (GitHub Actions: lint → test → build → push image → deploy compose stack), recorded demo. Deliverable: `docker compose up` → working live demo.

### Repository Layout
```
sepsis-ews/
├── data/
│   ├── raw/ (gitignored)
│   ├── simulator/          # replay generator for PhysioNet/MIMIC rows
├── pipeline/
│   ├── ingestion/          # Redpanda producer/consumer
│   ├── validation/         # Pydantic contracts
│   └── features/           # Polars transforms, Feast feature definitions
├── ml/
│   ├── training/           # offline pipeline, Optuna search, eval matrix
│   ├── online/              # River ADWIN model
│   ├── explainability/     # SHAP + DiCE
│   └── registry/           # MLflow client wrappers
├── monitoring/
│   ├── drift/               # Evidently + Alibi-Detect jobs
│   └── retrain_trigger/
├── api/
│   ├── main.py
│   ├── routers/
│   └── schemas/
├── dashboard/
├── infra/
│   ├── docker-compose.yml
│   └── github-actions/
├── tests/
├── Makefile
└── README.md
```

---

## Module 7: Portfolio Packaging

**README must include:** a Mermaid.js architecture diagram (data flow from simulator → broker → feature store → model → API → dashboard), a benchmark table (PR-AUC/F-beta/Brier/lead-time vs. a naive baseline), and a 60-second GIF or Loom of the live dashboard reacting to a simulated deterioration event — reviewers spend under 90 seconds on a README; the diagram and the table are what get screenshotted into a "worth a callback" pile.

**Resume bullets (tuned per role):**
- *ML Engineer:* "Built a real-time sepsis early-warning classifier (LightGBM + online drift-adaptive layer) achieving 0.71 PR-AUC under 6% base rate, with sub-50ms SHAP-based explainability per prediction."
- *Data Engineer:* "Designed a Kafka-compatible streaming feature pipeline (Redpanda → Polars → Feast/Redis) processing multi-resolution rolling clinical features with automated data-contract validation and dead-letter routing."
- *Backend Engineer:* "Shipped an async FastAPI inference service with WebSocket streaming, sub-100ms p95 latency, and a fully containerized 8-service Docker Compose deployment with CI/CD via GitHub Actions."

---

### The One Thing Worth Saying Out Loud

The dataset and the model are the least impressive part of this build — anyone can call `.fit()` on XGBoost. What actually separates this from a fresher project is the **closed loop**: drift detected → retrain triggered → challenger evaluated against the full cost matrix → promoted or rejected → redeployed, with every step versioned and explainable. Build that loop first if you're short on time, and let the model itself be "good enough" — a mediocre model in a real MLOps loop beats a great model in a notebook, every single time a reviewer who's actually shipped ML looks at it.
