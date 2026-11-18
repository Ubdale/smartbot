"""Shared request dependencies: the current user, and workspace-scoped queries.

Workspace isolation rule: every query for customer data must go through
`workspace_scoped(...)` with the current user's workspace. Rows outside the
caller's workspace are reported as 404, never 403, so their existence isn't leaked.
"""

from datetime import UTC, date, datetime
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import Select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.db import get_db
from app.core.security import InvalidTokenError, decode_access_token
from app.models import User, Workspace

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# Browsers authenticate with an HttpOnly cookie (the web app proxies /api to us, so
# the cookie belongs to the frontend's own domain). API clients keep using Bearer.
AUTH_COOKIE = "askdata_session"
# Cookie-authenticated writes must carry this header. Other sites can't add custom
# headers without a CORS preflight we never approve, so this blocks CSRF (on top of
# SameSite=Lax).
CSRF_HEADER = "X-AskData-CSRF"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    request: Request,
    bearer: Annotated[str | None, Depends(oauth2_scheme)],
    db: DbSession,
    settings: AppSettings,
) -> User:
    token = bearer or request.cookies.get(AUTH_COOKIE)
    if not token:
        raise _UNAUTHORIZED
    cookie_write = bearer is None and request.method not in _SAFE_METHODS
    if cookie_write and request.headers.get(CSRF_HEADER) != "1":
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing {CSRF_HEADER} header")
    try:
        claims = decode_access_token(settings, token)
    except InvalidTokenError:
        raise _UNAUTHORIZED from None

    user = db.get(User, claims.user_id)
    # Token must still match the user's current workspace (e.g. after a move or deletion).
    if user is None or user.workspace_id != claims.workspace_id:
        raise _UNAUTHORIZED
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def workspace_scoped[S: Select[Any]](stmt: S, model: Any, user: User) -> S:
    """Restrict a SELECT to rows of `model` in the user's workspace."""
    column = model.id if model is Workspace else model.workspace_id
    return stmt.where(column == user.workspace_id)


def not_found(what: str = "Resource") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} not found")


def get_today() -> date:
    """Today's date (UTC) for resolving "last quarter" etc. Overridden in tests."""
    return datetime.now(UTC).date()


Today = Annotated[date, Depends(get_today)]
