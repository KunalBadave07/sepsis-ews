# Sepsis-EWS — Full Development Execution Plan
### From Zero to Working Demo: Every Click, Every Command, Every File

This is a literal, no-assumptions walkthrough. If a step feels "too obvious to write down" — I wrote it down anyway. Follow it top to bottom. Each Sprint maps to Module 6 of the architecture spec.

---

## PHASE 0 — Machine Setup (Day 0, before any code)

### 0.1 Check what you already have
Open a terminal:
- **Windows:** press `Win` key, type `PowerShell`, hit Enter.
- **Mac:** press `Cmd + Space`, type `Terminal`, hit Enter.

Type these one at a time and press Enter after each:
```
python --version
git --version
docker --version
node --version
```
Whatever prints "not recognized" or "command not found" — install that one below. Skip anything that already shows a version number.

### 0.2 Install Python
1. Open your browser, go to `https://www.python.org/downloads/`
2. Click the big yellow "Download Python 3.x.x" button (get the latest 3.11 or 3.12 — avoid 3.13 if it's brand new, some ML libraries lag behind).
3. Run the installer. **Critical:** on the first install screen, check the box at the bottom that says **"Add python.exe to PATH"** before clicking Install. This is the single most common reason people's `python` command doesn't work afterward.
4. After install, reopen your terminal (close and reopen the window) and run `python --version` again to confirm.

### 0.3 Install Git
1. Go to `https://git-scm.com/downloads`
2. Download for your OS, run the installer, click "Next" through the defaults (defaults are fine for this project).
3. Confirm: `git --version` in terminal.
4. Set your identity (replace with your actual name/email):
```
git config --global user.name "Your Name"
git config --global user.email "you@email.com"
```

### 0.4 Install VS Code
1. Go to `https://code.visualstudio.com/`
2. Click "Download" — it auto-detects your OS.
3. Run the installer, accept defaults.
4. Open VS Code once installed. Click the Extensions icon on the left sidebar (looks like four squares).
5. Search and install these four extensions one at a time (type the name in the search box, click "Install"):
   - **Python** (by Microsoft)
   - **Docker** (by Microsoft)
   - **Pylance** (usually auto-installs with Python extension)
   - **Jupyter** (by Microsoft — for exploring data in notebooks)

### 0.5 Install Docker Desktop
1. Go to `https://www.docker.com/products/docker-desktop/`
2. Download for your OS.
3. Run the installer. On Windows, it may ask to enable WSL2 — click "Yes"/accept, it will prompt a restart. Restart if asked.
4. After install, **open the Docker Desktop application** (it must be running in the background — check for the whale icon in your system tray/menu bar) before any `docker` command will work.
5. Confirm in terminal: `docker run hello-world` — you should see a "Hello from Docker!" message.

### 0.6 Create accounts (do this now, not later when you're blocked)
- **GitHub** — `https://github.com/join` — you need this to host your code and for CI/CD.
- **PhysioNet** — `https://physionet.org/register/` — needed for the Sepsis Challenge dataset. Verify your email.
- **MLflow / Kaggle** — Kaggle account at `https://www.kaggle.com/account/login` (optional mirror source, PhysioNet is primary).

---

## PHASE 1 — Project Scaffolding (Day 1)

### 1.1 Create the project folder
Pick a real location, e.g. Documents. In terminal:
```
cd Documents
mkdir sepsis-ews
cd sepsis-ews
```

### 1.2 Open it in VS Code
Type in the same terminal:
```
code .
```
(The dot means "open current folder." VS Code opens with `sepsis-ews` as the root.)

### 1.3 Initialize git
In the VS Code terminal (open it via menu **Terminal → New Terminal**):
```
git init
```

### 1.4 Create the folder skeleton
Still in the terminal, run this exact block (creates every folder from the architecture spec in one shot):
```
mkdir data data/raw data/simulator pipeline pipeline/ingestion pipeline/validation pipeline/features ml ml/training ml/online ml/explainability ml/registry monitoring monitoring/drift monitoring/retrain_trigger api api/routers api/schemas dashboard infra infra/github-actions tests
```

### 1.5 Create a Python virtual environment
```
python -m venv venv
```
Activate it:
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
  - If you get a "scripts disabled" error, run this once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, type `Y`, then retry activation.
- **Mac/Linux:** `source venv/bin/activate`

You'll know it worked because your terminal prompt now starts with `(venv)`. **Every terminal session from now on, activate this first** before running any Python command.

### 1.6 Create requirements.txt
In VS Code's file explorer (left sidebar), right-click the root `sepsis-ews` folder → **New File** → name it `requirements.txt`. Paste this in:
```
pandas
polars
numpy
scikit-learn
lightgbm
xgboost
shap
river
optuna
mlflow
fastapi
uvicorn[standard]
pydantic
redis
feast
kafka-python
evidently
alibi-detect
streamlit
dvc
pytest
python-dotenv
```
Save (`Ctrl+S` / `Cmd+S`). Install everything:
```
pip install -r requirements.txt
```
This will take several minutes. If any single package fails, don't panic — comment it out with a `#` in front, install the rest, and troubleshoot that one package alone afterward (version conflicts are common with `alibi-detect` and `feast` specifically; install those two last, separately, if the combined install fails).

### 1.7 Create a `.gitignore`
New File → `.gitignore` in the root. Paste:
```
venv/
__pycache__/
*.pyc
data/raw/
.env
*.db
mlruns/
.DS_Store
```

### 1.8 First commit
```
git add .
git commit -m "Initial project scaffolding"
```

### 1.9 Create the GitHub repo and connect it
1. Go to `https://github.com/new`
2. Repository name: `sepsis-ews`. Leave "Initialize with README" **unchecked** (you already have local files). Click "Create repository."
3. GitHub shows you commands — copy the ones under "…or push an existing repository from the command line," they'll look like:
```
git remote add origin https://github.com/YOUR-USERNAME/sepsis-ews.git
git branch -M main
git push -u origin main
```
Run those in your terminal.

---

## PHASE 2 — Get the Data (Day 1-2)

### 2.1 Download the PhysioNet 2019 Sepsis dataset
1. Go to `https://physionet.org/content/challenge-2019/1.0.0/`
2. Log in with the account you made in 0.6.
3. Scroll to "Files" section, click **"Download the ZIP file"** (or use the individual `training_setA.zip` / `training_setB.zip` links — smaller, faster).
4. Save the ZIP to your Downloads folder.
5. Move it into your project: drag the ZIP file into `sepsis-ews/data/raw/` in your file explorer (Finder/Windows Explorer), or via terminal:
```
mv ~/Downloads/training_setA.zip data/raw/
```
6. Unzip it:
- **Mac:** double-click the ZIP.
- **Windows:** right-click → "Extract All."
- **Terminal (either OS):** `cd data/raw && unzip training_setA.zip && cd ../..`

You should now see a folder of `.psv` files (pipe-separated values — one per patient) inside `data/raw/`.

### 2.2 Sanity-check the data in a notebook
In VS Code, New File → `notebooks/explore.ipynb` (create the `notebooks` folder if VS Code doesn't auto-create it). Click "Select Kernel" top-right, choose your `venv` Python interpreter. First cell:
```python
import pandas as pd
df = pd.read_csv("../data/raw/training/p000001.psv", sep="|")
df.head()
```
Run the cell (Shift+Enter). You should see a table of hourly vitals. If this works, your data pipeline has something real to read.

---

## PHASE 3 — Sprint 1: Ingestion & Feature Store (Week 1-2)

### Sprint 1 Goal
Raw vitals in → validated, feature-engineered data in Redis, end to end.

### 3.1 Day-by-day breakdown
- **Day 1-2:** Redpanda + Pydantic validation
- **Day 3-4:** Polars feature transforms
- **Day 5-7:** Feast + Redis wiring
- **Day 8-10:** Integration test + buffer for bugs

### 3.2 Add Redpanda to your Docker setup
New File → `infra/docker-compose.yml`. Start with just this service (you'll add more each sprint):
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
      - --advertise-kafka-addr PLAINTEXT://localhost:9092
    ports:
      - "9092:9092"
  redis:
    image: redis:7
    ports:
      - "6379:6379"
```
Start it:
```
docker compose -f infra/docker-compose.yml up -d
```
Confirm both containers are running: `docker ps` — you should see `redpanda` and `redis` listed.

### 3.3 Write the Pydantic validation contract
New File → `pipeline/validation/schema.py`. Paste the `VitalReading` class from the architecture spec (Module 2) exactly as written there.

### 3.4 Write the simulator
New File → `data/simulator/replay.py`. This script should read `.psv` files row by row and publish each row as a JSON message to the `vitals.raw` Kafka topic on a short delay, to simulate a live stream. Start simple — get one patient file replaying before generalizing to all of them.

### 3.5 Write the validation consumer
New File → `pipeline/ingestion/consumer.py`. Consumes from `vitals.raw`, validates each message against `VitalReading`, publishes valid rows to `vitals.clean` and invalid ones to `vitals.dlq`.

### 3.6 Test it end to end
Open **three separate terminal tabs** (all with `venv` activated):
- Tab 1: `python pipeline/ingestion/consumer.py`
- Tab 2: `python data/simulator/replay.py`
- Tab 3: watch the topic — install `kcat` or just print consumer output to confirm messages flow.

If you see clean messages printing in Tab 1, Sprint 1's core loop works. Commit:
```
git add .
git commit -m "Sprint 1: ingestion + validation working end to end"
git push
```

*(Feature transforms and Feast wiring follow the same pattern: write the code file, test it standalone, wire it into the consumer chain, commit. Repeat this rhythm for every sprint below — I'm not going to keep re-explaining "open terminal, run file, commit" after this point, just apply it.)*

---

## PHASE 4 — Sprint 2: Core Model & Explainability (Week 3-4)

### 4.1 Training script
New File → `ml/training/train.py`. Loads the validated/feature-engineered data, splits by patient (never split by row — that leaks data across time for the same patient), trains LightGBM with Optuna hyperparameter search, logs everything to MLflow.

### 4.2 Start MLflow locally
```
mlflow ui --port 5000
```
Open `http://localhost:5000` in your browser to watch runs appear as you train.

### 4.3 SHAP explainability
New File → `ml/explainability/shap_pipeline.py`. Compute the K-means-summarized background set, cache it (pickle or joblib to `ml/registry/`), write a function that returns top-5 SHAP features for a given row in under 50ms — time it with Python's `time.perf_counter()` before/after and print it, don't just assume it's fast.

---

## PHASE 5 — Sprint 3: Drift Engine, SWADT & API (Week 5-6)

### 5.1 River online model
New File → `ml/online/adwin_model.py`.

### 5.2 SWADT implementation
New File → `monitoring/retrain_trigger/swadt.py`. Implement the exact formulas from Section 4 of the SWADT paper — `U_d(t)`, `T(t)`, `τ(t)` — as literal Python functions. This is the part of the whole project that's actually yours; don't rush it.

### 5.3 FastAPI service
New File → `api/main.py` + `api/routers/predict.py`. Run locally to test before containerizing:
```
uvicorn api.main:app --reload --port 8000
```
Open `http://localhost:8000/docs` — FastAPI auto-generates an interactive test page. Use it to send a test prediction request before writing any dashboard code.

---

## PHASE 6 — Sprint 4: Full Containerization & Dashboard (Week 7-8)

### 6.1 Expand docker-compose.yml
Add the `postgres`, `mlflow-server`, `feature-pipeline`, `inference-api`, `drift-monitor`, and `dashboard` services to the same file from Section 3.2, following the topology in the architecture spec's Module 5.

### 6.2 Dashboard background image (the "silly" step you asked about)
If you want a background/hero image for the Streamlit or Next.js dashboard:
1. Go to `https://unsplash.com` (free, no-attribution-required stock images).
2. Search something like "hospital monitor" or "data dashboard dark."
3. Click an image → click the down-arrow **Download** button → choose "Small" or "Medium" size (you don't need a 6000px image for a web background).
4. Move the downloaded file into a new folder: `dashboard/assets/` (create this folder first).
5. Reference it in your dashboard code by relative path, e.g. `dashboard/assets/hero.jpg`.

Don't use random Google Image Search results for anything you might ever show publicly — no license clarity. Unsplash/Pexels/Pixabay are free-to-use and won't come back to bite you.

### 6.3 Full stack up
```
docker compose -f infra/docker-compose.yml up --build
```
Open the dashboard URL it prints (typically `http://localhost:8501` for Streamlit).

### 6.4 GitHub Actions CI
New File → `.github/workflows/ci.yml` (note: this exact path, GitHub only recognizes workflows in `.github/workflows/`). Minimal starter:
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
      - run: pytest tests/
```
Commit and push — check the "Actions" tab on your GitHub repo page to watch it run.

---

## PHASE 7 — Wrap-Up & Demo Packaging

### 7.1 Write the README
New File → `README.md` in the root. Include: one-paragraph pitch, architecture diagram (use Mermaid.js code blocks — GitHub renders these natively, no image export needed), the benchmark table once you have real numbers, and the exact `docker compose up` quickstart command.

### 7.2 Record the demo
- **Mac:** Cmd+Shift+5 opens the built-in screen recorder.
- **Windows:** Win+G opens Xbox Game Bar's recorder.
- Keep it under 90 seconds: show the simulator pushing data, the dashboard updating live, click one SHAP waterfall, done.
- Upload the file to a Loom account (`https://www.loom.com`, free tier) or just attach the raw video/GIF in your GitHub README.

### 7.3 Final push
```
git add .
git commit -m "v1.0 — full stack working demo"
git tag v1.0
git push origin main --tags
```

---

## A Note on Pacing

Don't try to do this in a weekend. Eight weeks across four sprints, roughly 8-10 focused hours a week, is realistic for one person building this solo alongside a job or classes. If you rush Sprint 1's foundation to get to the "impressive" ML part faster, you'll spend Sprint 3 fighting plumbing bugs instead of building SWADT — the boring infrastructure work early is what makes the interesting work later actually work.
