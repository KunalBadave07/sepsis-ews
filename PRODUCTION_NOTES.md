# Production Notes — What's Real, What's Simulated, What's Required

This file exists so nobody — a recruiter, a senior engineer, or future-me —
overstates what this project is. Read it before presenting this system to
anyone as more than a portfolio prototype.

## What's genuinely real in this build
- The streaming ingestion, validation, and feature engineering pipeline
  runs end to end against real (public, de-identified) clinical data.
- The offline classifier is trained and evaluated with honest metrics
  (PR-AUC, F-beta, Brier score — not just accuracy, which is misleading
  under this class imbalance).
- SHAP-based explainability is real and measured at sub-50ms per prediction.
- SWADT (the novel drift-trigger mechanism) is implemented and validated
  via a differential test proving importance-weighted drift detection.
- The API enforces human-in-the-loop review, structurally cannot return
  a treatment/medication recommendation, and logs a full predict-to-
  acknowledgment audit trail.

## What's simulated, not real
- All patient data is public retrospective research data (PhysioNet 2019
  Sepsis Challenge / MIMIC-IV), not live PHI.
- The FHIR adapter demonstrates the correct integration pattern; it is
  not connected to any real EHR system (Epic, Cerner, etc.).
- Authentication uses a hardcoded demo user store, not a real hospital
  identity provider / SSO.
- The online drift-sensing model uses a proxy labeling heuristic in
  shadow mode, not real delayed clinical ground truth.
- "24/7 availability" is a documented target architecture, not a
  measured production SLA — this runs in Docker Compose on a laptop.

## What real clinical deployment would additionally require
- FDA clearance as Software as a Medical Device (most likely Class II),
  including a formal quality management system under IEC 62304.
- A prospective clinical validation trial with IRB approval — retrospective
  benchmark performance does not establish real-world clinical validity.
- HIPAA risk assessment, likely HITRUST or SOC 2 certification, and a
  signed Business Associate Agreement with any deploying hospital.
- Real EHR integration contracts and engineering (HL7 FHIR/v2 with Epic,
  Cerner, or similar), owned jointly with the hospital's IT department.
- A dedicated on-call/SRE process for genuine 24/7 availability guarantees.
- Legal and regulatory review of every claim made about clinical impact
  before any such claim is made publicly.

## The honest one-sentence summary
This is a well-engineered technical prototype demonstrating production-
grade ML system design discipline — streaming architecture, explainability,
adaptive drift detection, and safety guardrails — built on public research
data. It is not clinically validated, not FDA-cleared, and not deployed
in any real hospital. Anyone asking should get this exact answer.