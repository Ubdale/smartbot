"""Password hashing (bcrypt) and access tokens (JWT, HS256)."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.core.config import Settings

JWT_ALGORITHM = "HS256"
TOKEN_TYPE = "access"  # noqa: S105 - token type label, not a secret

# bcrypt only looks at the first 72 bytes; we reject longer passwords instead of truncating.
BCRYPT_MAX_BYTES = 72

# Used to burn the same CPU time when the email doesn't exist (no user enumeration by timing).
_DUMMY_HASH = bcrypt.hashpw(b"timing-equaliser", bcrypt.gensalt()).decode()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str | None) -> bool:
    candidate = password.encode()
    if password_hash is None:
        bcrypt.checkpw(candidate[:BCRYPT_MAX_BYTES], _DUMMY_HASH.encode())
        return False
    if len(candidate) > BCRYPT_MAX_BYTES:
        return False
    return bcrypt.checkpw(candidate, password_hash.encode())


@dataclass(frozen=True)
class TokenClaims:
    user_id: str
    workspace_id: str


class InvalidTokenError(Exception):
    pass


def create_access_token(
    settings: Settings, *, user_id: str, workspace_id: str, now: datetime | None = None
) -> str:
    issued = now or datetime.now(UTC)
    payload = {
        "sub": user_id,
        "ws": workspace_id,
        "type": TOKEN_TYPE,
        "iat": issued,
        "exp": issued + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=JWT_ALGORITHM)


def decode_access_token(settings: Settings, token: str) -> TokenClaims:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[JWT_ALGORITHM],  # never accept "none" or other algorithms
            options={"require": ["sub", "ws", "exp", "iat", "type"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc

    sub, ws = payload.get("sub"), payload.get("ws")
    if payload.get("type") != TOKEN_TYPE or not isinstance(sub, str) or not isinstance(ws, str):
        raise InvalidTokenError("malformed token claims")
    return TokenClaims(user_id=sub, workspace_id=ws)
