# api/routers/auth.py
from fastapi import APIRouter, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from api.security.auth import FAKE_USERS, create_token

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_token(form_data.username, user["role"])
    return {"access_token": token, "token_type": "bearer"}