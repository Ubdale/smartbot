"""Shared request dependencies: the current user, and workspace-scoped queries.

Workspace isolation rule: every query for customer data must go through
`workspace_scoped(...)` with the current user's workspace. Rows outside the
caller's workspace are reported as 404, never 403, so their existence isn't leaked.
"""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import Select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.db import get_db
from app.core.security import InvalidTokenError, decode_access_token
from app.models import User, Workspace

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    db: DbSession,
    settings: AppSettings,
) -> User:
    if not token:
        raise _UNAUTHORIZED
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
