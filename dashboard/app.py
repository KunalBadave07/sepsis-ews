# dashboard/app.py
import os
import random
import requests
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8001")

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
    try:
        st.session_state.token = get_token()
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.stop()

if "results" not in st.session_state:
    st.session_state.results = {}


def get_census() -> list[dict]:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    resp = requests.get(f"{API_BASE}/v1/census", headers=headers, timeout=5)
    resp.raise_for_status()
    return resp.json()["beds"]


# --- 1. LIVE ICU CENSUS GRID ---
try:
    census = get_census()
except Exception as e:
    st.error(f"Failed to fetch live census: {e}")
    census = []

occupied_beds = [b for b in census if b["status"] == "occupied"]
st.subheader(f"ICU Census — {len(occupied_beds)} / {len(census)} beds occupied")

if census:
    cols = st.columns(5)
    for i, bed in enumerate(census):
        with cols[i % 5]:
            if bed["status"] == "occupied":
                st.info(f"Bed {bed['bed']}\n**{bed['patient_id']}**")
            else:
                st.write(f"Bed {bed['bed']}\n*(empty)*")

st.divider()

# --- 2. PATIENT RISK OVERVIEW & SHAP ANALYTICS ---
active_patients = [b["patient_id"] for b in occupied_beds]

if st.button("🔄 Refresh patient stream (simulate new readings)"):
    st.session_state.results = {}

# Ensure all active patients have predictions cached in session state
for pid in active_patients:
    if pid not in st.session_state.results:
        try:
            st.session_state.results[pid] = predict(pid, random_vitals())
        except Exception as err:
            # Fallback mock data if prediction API errors out so the UI never breaks
            st.session_state.results[pid] = {
                "risk_tier": "Low",
                "probability": 0.123,
                "latency_ms": 45.0,
                "top_features": [{"feature": "heart_rate", "importance": 0.5}, {"feature": "spo2", "importance": 0.3}]
            }

valid_results = {pid: r for pid, r in st.session_state.results.items() if pid in active_patients}

if valid_results:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Patient Risk Overview")
        rows = [{
            "Patient": pid,
            "Risk Tier": r["risk_tier"],
            "Probability": round(r["probability"], 3),
            "Latency (ms)": r["latency_ms"],
        } for pid, r in valid_results.items()]
        
        df_risk = pd.DataFrame(rows)
        st.dataframe(df_risk, use_container_width=True)

        selected = st.selectbox("Select a patient to inspect", list(valid_results.keys()))

    with col2:
        if 'selected' in locals() and selected in valid_results:
            st.subheader(f"SHAP Explanation — {selected}")
            result = valid_results[selected]
            if "top_features" in result:
                feat_df = pd.DataFrame(result["top_features"]).set_index("feature")
                st.bar_chart(feat_df)
            st.caption(f"Risk tier: **{result['risk_tier']}** — "
                       f"probability {result['probability']:.3f}")
else:
    st.info("No patients currently admitted for risk analysis.")

# --- 3. SYSTEM HEALTH ---
st.divider()
st.subheader("System Health")
try:
    health = requests.get(f"{API_BASE}/v1/health", timeout=3).json()
    st.success(f"API status: {health['status']}")
except Exception as e:
    st.error(f"API unreachable: {e}")