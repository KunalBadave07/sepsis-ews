# api/security/auth.py
"""
Minimal JWT auth for the demo. NOTE: hardcoded demo users and a
hardcoded secret key are ONLY acceptable because this is a local
prototype. Real deployment would use a real identity provider (hospital
SSO) — this is flagged again in PRODUCTION_NOTES.md.
"""
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "dev-only-secret-never-use-in-real-deployment"
ALGORITHM = "HS256"

# demo user store: username -> (password, role)
FAKE_USERS = {
    "nurse_jane": {"password": "demo123", "role": "clinician"},
    "admin_sam": {"password": "demo123", "role": "admin"},
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")


def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=8),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                             detail="Invalid or expired token")


def require_role(*allowed_roles: str):
    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                 detail=f"Requires one of roles: {allowed_roles}")
        return user
    return checker