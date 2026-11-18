from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, not_found, workspace_scoped
from app.models import User, Workspace
from app.schemas.auth import UserOut, WorkspaceOut

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("/me", response_model=WorkspaceOut)
def my_workspace(user: CurrentUser, db: DbSession) -> Workspace:
    ws = db.scalar(workspace_scoped(select(Workspace), Workspace, user))
    if ws is None:
        raise not_found("Workspace")
    return ws


@router.get("/me/members", response_model=list[UserOut])
def my_members(user: CurrentUser, db: DbSession) -> list[User]:
    stmt = workspace_scoped(select(User), User, user).order_by(User.created_at)
    return list(db.scalars(stmt))


@router.get("/{workspace_id}", response_model=WorkspaceOut)
def get_workspace(workspace_id: str, user: CurrentUser, db: DbSession) -> Workspace:
    stmt = workspace_scoped(select(Workspace).where(Workspace.id == workspace_id), Workspace, user)
    ws = db.scalar(stmt)
    if ws is None:
        raise not_found("Workspace")
    return ws
