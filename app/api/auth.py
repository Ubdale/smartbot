from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.deps import AUTH_COOKIE, AppSettings, CurrentUser, DbSession
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User, Workspace
from app.schemas.auth import SignupRequest, SignupResponse, TokenResponse, UserOut, WorkspaceOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(settings: AppSettings, user: User, response: Response) -> TokenResponse:
    token = create_access_token(settings, user_id=user.id, workspace_id=user.workspace_id)
    max_age = settings.access_token_expire_minutes * 60
    response.set_cookie(
        AUTH_COOKIE,
        token,
        max_age=max_age,
        httponly=True,  # never readable by page JavaScript
        secure=settings.app_env == "prod",
        samesite="lax",
        path="/",
    )
    return TokenResponse(access_token=token, expires_in=max_age)


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(
    body: SignupRequest, response: Response, db: DbSession, settings: AppSettings
) -> SignupResponse:
    """Create a company workspace with the caller as its owner."""
    if db.scalar(select(User.id).where(User.email == body.email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")

    workspace = Workspace(name=body.workspace_name)
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        role="owner",
        workspace=workspace,
    )
    db.add_all([workspace, user])
    try:
        db.commit()
    except IntegrityError:  # concurrent signup with the same email
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "An account with this email already exists"
        ) from None

    token = _token_response(settings, user, response)
    return SignupResponse(
        **token.model_dump(),
        user=UserOut.model_validate(user),
        workspace=WorkspaceOut.model_validate(workspace),
    )


@router.post("/login", response_model=TokenResponse)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    response: Response,
    db: DbSession,
    settings: AppSettings,
) -> TokenResponse:
    """OAuth2 password flow: send `username` (your email) and `password` as form fields."""
    user = db.scalar(select(User).where(User.email == form.username.strip().lower()))
    # Always run bcrypt (against a dummy hash if no such user) so timing doesn't reveal emails.
    password_ok = verify_password(form.password, user.password_hash if user else None)
    if user is None or not password_ok:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_response(settings, user, response)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, settings: AppSettings) -> None:
    response.delete_cookie(
        AUTH_COOKIE, path="/", httponly=True, secure=settings.app_env == "prod", samesite="lax"
    )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user
