# Sepsis-EWS — Clinical Deployment Framework
### Honest Answers to Enterprise/Clinical Constraints, and What They Mean for Sprint 3 Onward

This document exists so nobody — including future-you in an interview — overstates what this system is. Every answer below distinguishes **"what we design/simulate as a portfolio demonstration"** from **"what real hospital deployment would actually require."** Conflating the two is the single fastest way to lose credibility with a technical or clinical interviewer.

---

## 1. The Experience Gap
**Question:** Why would an experienced physician use this over their own judgment?

**Honest answer:** They wouldn't use *this build* — it's a demo. But the *design principle* it demonstrates is real and already industry-accepted: early warning scores (NEWS2, qSOFA, MEWS) exist precisely because human clinicians, however expert, cannot continuously hold 40+ variables across a dozen patients in working memory while also doing everything else their shift requires. The value proposition is never "the AI knows more than the doctor" — it's "the AI never gets tired, never gets distracted mid-shift, and notices a slow multivariate trend across 6 hours that a human glancing at one vitals chart at a time will miss." Position this as **augmentation of existing early-warning-score practice**, not replacement of clinical judgment. Say this explicitly in your README and in interviews — it's both true and the correct framing.

## 2. Availability (24/7 Uptime)
**Honest answer:** Your local Docker Compose stack is a **development/demo environment**, not a production system, and should never be described as having uptime guarantees. Real hospital-grade availability requires redundant multi-zone deployment, automatic failover, health-checked container orchestration (Kubernetes, not Compose), and a defined SLA with an on-call rotation behind it. **What Sprint 3 will do:** design and document the target production architecture (redundancy, health checks, graceful degradation when a dependency like Redis is unreachable) as a documented "Production Availability Design" — built for real, running locally, but explicitly labeled as a design specification for what a production rollout would need, not a claim that this demo has 24/7 uptime.

## 3. Clinical Validity
**Honest answer: no.** This system has not undergone, and a portfolio project by definition cannot undergo, prospective clinical validation. Its "accuracy" numbers come from a retrospective public research dataset, not a live clinical trial comparing outcomes against standard of care. It does not provide "real treatment analysis" — it provides a **risk-stratification score with feature attribution**, full stop. Never claim clinical validity beyond that. Your README should state directly: *"Trained and evaluated on public retrospective research data (PhysioNet 2019 Challenge / MIMIC-IV). Not clinically validated. Not a diagnostic device."*

## 4. Critical Impact ("Can it save a life?")
**Honest answer:** Not as built, and you should never say yes to this question in an interview or on a resume. The correct, defensible answer is: *"The class of system this demonstrates — early deterioration warning with explainable risk scoring — has published clinical literature showing outcome improvements when integrated into hospital workflows with appropriate human oversight. This specific implementation is a research/portfolio prototype and has not been validated to make that claim itself."* That's a real, honest, still-impressive answer. Overclaiming here is both dishonest and, if said to the wrong person, a liability red flag about your judgment.

## 5. Portability
**Honest answer:** The dashboard (once built in Sprint 4) will be a responsive web app, accessible from any device with a browser — that part is genuinely achievable and worth building well. Real hospital portability (SSO across hospital-issued devices, integration with existing clinical workstations, badge-tap login) is infrastructure the hospital's IT department owns, not something a standalone project provides. **Design decision for Sprint 3 onward:** build the API with token-based auth (JWT) from day one, so the *pattern* of "any authenticated device can pull the same live data" is real, even though full enterprise SSO integration is out of scope.

## 6. Scope of Advice — This Is a Hard Architectural Rule, Not a Suggestion
**The system will never, at any point, output a treatment, medication, or dosage recommendation.** This isn't a feature gap to fill later — it's a permanent boundary. Recommending treatment is practicing medicine, and getting this wrong is both an ethical failure and a legal one (regardless of your intent, or a user's stated "just for research" framing). Sprint 3 will enforce this at the API contract level: the response schema **structurally cannot contain a treatment/medication field** — not "we chose not to fill it in," but "the schema doesn't have a slot for it." This is a design choice worth being proud of and stating explicitly in your README as a safety feature, not a limitation.

## 7. End-User Daily Workflow
**Honest answer:** In the demo, the "workflow" is: dashboard shows a live-updating list of monitored patients, sorted by risk tier; a nurse or physician glances at it during rounds; clicking a patient shows the SHAP explanation for *why* they're flagged. That's a real, demonstrable workflow — build it well in Sprint 4. What it does **not** do: replace charting, integrate into order entry, or reduce documentation burden — don't imply it does more than it does.

## 8. AI Explainability (Trust)
**Already real, already built (Sprint 2).** Every prediction ships with its top-5 SHAP feature attributions, computed sub-50ms. This is your strongest, most legitimate answer to "black box" concerns — lean on it honestly, it's the one claim in this whole list you've actually earned.

## 9. Latency in Critical Care
**Honest answer:** Your explainability computation is sub-50ms (verified in Sprint 2). That is **not** the same as an end-to-end guaranteed round-trip SLA in a real critical-care environment, which would also need to account for network latency, EHR query time, and failover behavior — none of which exist in a local demo. **Sprint 3 deliverable:** document a target SLA (e.g., "sub-second p99 end-to-end") as a design spec, and build the FastAPI service to actually measure and log its own real latency from request to response — so the number you eventually report is measured, not asserted.

## 10. Data Integration with Legacy EHR Systems
**Honest answer:** Real EHR integration means HL7 FHIR or HL7v2 interfaces with Epic, Cerner, or similar — systems this project will never actually connect to, and claiming otherwise is an easy claim to get caught overstating. **What we can legitimately build:** an ingestion adapter that accepts data shaped like a **FHIR `Observation` resource** (the real interoperability standard) and translates it into your internal `VitalReading` schema. This demonstrates you understand the real integration pattern hospitals use, without pretending you've integrated with a real EHR. Label it clearly: *"FHIR-shaped adapter, simulated upstream source — not connected to a live EHR."*

## 11. Liability & Guardrails / Human-in-the-Loop
**This is a required architectural layer, not optional polish.** Sprint 3 will add:
- Every prediction is a **flag for review**, never an autonomous action — the system cannot page a rapid-response team or alter a chart by itself.
- An **acknowledgment endpoint**: a clinician (simulated user role) must explicitly acknowledge or dismiss each alert; this acknowledgment is logged.
- A **full audit log** of every prediction, its explanation, and its human disposition — because in any real regulated environment, "what did the model say and what did the human do about it" must be reconstructable after the fact.

## 12. Data Privacy
**Honest answer:** No real PHI (Protected Health Information) exists in this project — PhysioNet/MIMIC data is de-identified public research data under a data use agreement, not live patient data. That said, build the *pattern* correctly, because it's what a real deployment requires and it's a genuinely strong resume line: encryption in transit (TLS) and at rest, role-based access control, and audit logging of all data access — not because HIPAA currently applies to a public dataset, but because building the habit and the architecture now is exactly what a real deployment would need on day one, and you should be able to say "I designed for HIPAA-aligned controls even though this demo doesn't legally require them" in an interview.

---

## What This Means for Sprint 3, Concretely

Sprint 3 was originally: River/ADWIN online model, SWADT, FastAPI serving layer. All of that stays — but four new deliverables get added, because they're the direct engineering answer to the questions above:

| New Deliverable | Answers Which Question(s) |
|---|---|
| API response schema with **no treatment/dosage field, ever** | Scope of Advice (#6) |
| JWT-based auth + basic role scaffold (clinician/admin/viewer) | Portability (#5), Data Privacy (#12) |
| **Acknowledgment endpoint** + full audit log of prediction → human disposition | Liability & Guardrails (#11) |
| FHIR-shaped ingestion adapter (simulated, clearly labeled) | Data Integration (#10) |
| Measured (not asserted) end-to-end latency logging | Latency (#9) |
| `PRODUCTION_NOTES.md` — one file, honestly separating "what's real here" from "what real deployment requires" | All of the above, at once |

The core ML/drift work (River, SWADT, SHAP-serving) doesn't change — it gets wrapped in the guardrails a real clinical system would need, and every one of those guardrails gets documented honestly rather than oversold.

---

## Sprint 3 — Technical Milestones & Implementation Plan

**Sprint Goal:** Online drift-sensing model + SWADT retraining trigger + guardrailed FastAPI serving layer, with human-in-the-loop, audit logging, and honest production-readiness documentation, all working end to end.

### Milestone 1 (Day 1-2): River Online Model (Drift Sensor)
- Implement `AdaptiveRandomForestClassifier` wrapped with `ADWIN`, running in shadow mode against the same feature stream from Sprint 1.
- Deliverable: a script that logs when ADWIN detects a change point, using synthetic drift injection (deliberately shift a feature's distribution mid-stream) to prove it actually fires, not just trust that it should.

### Milestone 2 (Day 3-5): SWADT Implementation
- Implement the exact formulas from the SWADT paper (`U_d(t)`, `T(t)`, adaptive `τ(t)`) as real Python functions consuming ADWIN's drift signal + the SHAP pipeline's live feature importances from Sprint 2.
- Deliverable: a test that proves SWADT suppresses a trigger for a drifting-but-unimportant feature while firing for a drifting-and-important one — this is the one test that actually validates your whole thesis, don't skip it.

### Milestone 3 (Day 6-8): Guardrailed FastAPI Serving Layer
- `POST /v1/predict` — returns probability + SHAP top features + model version. Response schema has no treatment/medication field by design.
- `POST /v1/predict/{prediction_id}/acknowledge` — simulated clinician acknowledgment, required for the audit trail.
- JWT auth middleware with role scaffold.
- Every request/response logged to an append-only audit table (SQLite is fine locally) with timestamp, prediction, explanation, and eventual human disposition.
- Latency measured and logged on every request — real numbers, not claims.

### Milestone 4 (Day 9-10): FHIR-Shaped Adapter + Production Notes
- A small translation layer accepting a FHIR-`Observation`-shaped JSON payload and mapping it to your internal `VitalReading` schema — proves you understand the real interoperability pattern without claiming real EHR integration.
- Write `PRODUCTION_NOTES.md`: a short, direct document listing exactly what's real, what's simulated, and what real deployment would additionally require (FDA clearance, clinical trial, HIPAA/HITRUST audit, EHR integration contracts). This file is your credibility insurance — it's what you show a skeptical interviewer who asks "wait, is this actually used in a hospital?"

### Definition of Done for Sprint 3
- [ ] ADWIN detects a synthetic drift injection and logs a change point
- [ ] SWADT's urgency score and adaptive threshold are implemented and unit-tested against both a "drifting-unimportant" and "drifting-important" synthetic scenario
- [ ] `/v1/predict` never returns a treatment/dosage field — verified by a test, not just by eye
- [ ] Every prediction is logged with a full audit trail including eventual acknowledgment
- [ ] Latency is measured and logged per-request, not asserted
- [ ] `PRODUCTION_NOTES.md` exists and honestly separates demo reality from production requirements

Ping me when you're ready and I'll walk you through Milestone 1 line by line, same as the last two sprints.
