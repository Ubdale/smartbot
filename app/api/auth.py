from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app import audit
from app.api.deps import AUTH_COOKIE, AppSettings, CurrentUser, DbSession
from app.api.team import hash_token
from app.core.security import BCRYPT_MAX_BYTES, create_access_token, hash_password, verify_password
from app.models import Invite, User, Workspace
from app.schemas.auth import SignupRequest, SignupResponse, TokenResponse, UserOut, WorkspaceOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(settings: AppSettings, user: User, response: Response) -> TokenResponse:
    token = create_access_token(
        settings,
        user_id=user.id,
        workspace_id=user.workspace_id,
        token_version=user.token_version or 0,
    )
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

    audit.record(db, user, "workspace.created", target_type="workspace",
                 target_id=workspace.id)  # fmt: skip
    db.commit()
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
    audit.record(db, user, "user.login", target_type="user", target_id=user.id)
    db.commit()
    return _token_response(settings, user, response)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, settings: AppSettings) -> None:
    response.delete_cookie(
        AUTH_COOKIE, path="/", httponly=True, secure=settings.app_env == "prod", samesite="lax"
    )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user


# --- invitations, password change, sign out everywhere -----------------------------------


class InviteInfo(BaseModel):
    email: str
    role: str
    workspace_name: str


def _fits_bcrypt(v: str) -> str:
    if len(v.encode()) > BCRYPT_MAX_BYTES:
        raise ValueError(f"password must be at most {BCRYPT_MAX_BYTES} bytes")
    return v


class AcceptInvite(BaseModel):
    token: str = Field(min_length=10, max_length=200)
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def _check(cls, v: str) -> str:
        return _fits_bcrypt(v)


class ChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

    @field_validator("new_password")
    @classmethod
    def _check(cls, v: str) -> str:
        return _fits_bcrypt(v)


def _open_invite(db: DbSession, token: str) -> Invite:
    invite = db.scalar(select(Invite).where(Invite.token_hash == hash_token(token)))
    if invite is None or invite.accepted_at is not None or invite.expires_at <= datetime.now(UTC):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "This invite link is invalid or has expired."
        )
    return invite


@router.get("/invites/{token}", response_model=InviteInfo)
def invite_info(token: str, db: DbSession) -> InviteInfo:
    """What the invite page shows before someone sets their password."""
    invite = _open_invite(db, token)
    workspace = db.get(Workspace, invite.workspace_id)
    return InviteInfo(
        email=invite.email, role=invite.role, workspace_name=workspace.name if workspace else ""
    )


@router.post("/invites/accept", response_model=TokenResponse)
def accept_invite(
    body: AcceptInvite, response: Response, db: DbSession, settings: AppSettings
) -> TokenResponse:
    invite = _open_invite(db, body.token)
    if db.scalar(select(User.id).where(User.email == invite.email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(
        email=invite.email,
        password_hash=hash_password(body.password),
        role=invite.role,
        workspace_id=invite.workspace_id,
    )
    invite.accepted_at = datetime.now(UTC)
    db.add(user)
    db.flush()
    audit.record(db, user, "invite.accepted", target_type="invite", target_id=invite.id)
    db.commit()
    return _token_response(settings, user, response)


@router.post("/change-password", response_model=TokenResponse)
def change_password(
    body: ChangePassword,
    user: CurrentUser,
    response: Response,
    db: DbSession,
    settings: AppSettings,
) -> TokenResponse:
    """New password; every other session (other browsers, stolen tokens) is signed out."""
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Your current password is wrong.")
    user.password_hash = hash_password(body.new_password)
    user.token_version = (user.token_version or 0) + 1
    audit.record(db, user, "password.changed", target_type="user", target_id=user.id)
    db.commit()
    return _token_response(settings, user, response)


@router.post("/logout-everywhere", status_code=status.HTTP_204_NO_CONTENT)
def logout_everywhere(
    user: CurrentUser, response: Response, db: DbSession, settings: AppSettings
) -> None:
    user.token_version = (user.token_version or 0) + 1
    audit.record(db, user, "sessions.revoked", target_type="user", target_id=user.id)
    db.commit()
    logout(response, settings)
