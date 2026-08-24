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