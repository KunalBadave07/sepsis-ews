# api/main.py
from fastapi import FastAPI, Depends, HTTPException
from monitoring.audit.audit_log import init_db
from api.security.auth import get_current_user
from pipeline.census.factory import get_default_manager
from api.routers import auth, predict

app = FastAPI(title="Sepsis-EWS API", version="0.1.0")

init_db()

app.include_router(auth.router)
app.include_router(predict.router)

_manager = None
def get_manager():
    global _manager
    if _manager is None:
        _manager = get_default_manager(total_beds=20)
    return _manager

@app.get("/v1/census")
def get_census(user: dict = Depends(get_current_user)):
    manager = get_manager()
    return {"beds": manager.get_census()}

@app.get("/v1/health")
def health():
    return {"status": "ok"}