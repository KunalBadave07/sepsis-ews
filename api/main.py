# api/main.py
from fastapi import FastAPI
from api.routers import predict, auth
from monitoring.audit.audit_log import init_db

app = FastAPI(title="Sepsis-EWS API", version="0.1.0")

init_db()

app.include_router(auth.router)
app.include_router(predict.router)


@app.get("/v1/health")
def health():
    return {"status": "ok"}