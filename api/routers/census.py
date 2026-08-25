# api/routers/census.py
from fastapi import APIRouter, Depends, HTTPException
from api.security.auth import get_current_user
from pipeline.census.factory import get_default_manager

router = APIRouter(tags=["census"])
_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = get_default_manager(total_beds=20)
    return _manager

@router.get("/census")
def get_census(user: dict = Depends(get_current_user)):
    manager = get_manager()
    return {"beds": manager.get_census()}