"""Dashboards: pin answers as tiles and view them re-run on the latest data."""

from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.datasets import require_parquet
from app.api.deps import AppSettings, CurrentUser, DbSession, Today, not_found
from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.factory import get_llm
from app.models import Conversation, Dashboard, Dataset, Message, User
from app.query.executor import (
    columns_from_profile,
    execute_plan,
    validate_plan,
)
from app.query.presentation import choose_chart, describe_plan, fallback_summary
from app.query.retrieval import apply_retrieval
from app.schemas.query_plan import QueryPlan

router = APIRouter(prefix="/dashboards", tags=["dashboards"])
MAX_TILES = 20
TILE_ROWS = 50


class DashboardCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PinIn(BaseModel):
    message_id: str
    dashboard_id: str | None = None  # pin to this dashboard, else to `name` (new)
    name: str | None = Field(default=None, max_length=200)


class DashboardSummary(BaseModel):
    id: str
    name: str
    tiles: int


class TileOut(BaseModel):
    question: str
    dataset_id: str
    dataset_name: str
    summary: str
    columns: list[str]
    rows: list[list[Any]]
    chart: dict[str, Any] | None
    error: str | None = None


class DashboardView(BaseModel):
    id: str
    name: str
    tiles: list[TileOut]


def _own(db: DbSession, user: User, dashboard_id: str) -> Dashboard:
    d = db.get(Dashboard, dashboard_id)
    if d is None or d.user_id != user.id or d.workspace_id != user.workspace_id:
        raise not_found("Dashboard")
    return d


def _render_tile(
    settings: Settings, db: DbSession, tile: dict[str, Any], llm: LLMProvider, today: date
) -> TileOut:
    ds = db.get(Dataset, tile["dataset_id"])
    base = TileOut(question=tile["question"], dataset_id=tile["dataset_id"],
                   dataset_name=ds.name if ds else "", summary="", columns=[], rows=[],
                   chart=None)  # fmt: skip
    if ds is None:
        base.error = "The data source was deleted."
        return base
    try:
        parquet = require_parquet(settings, ds)
        vp = validate_plan(QueryPlan.model_validate(tile["plan"]),
                           columns_from_profile(ds.columns), today)  # fmt: skip
        if any(f.op == "about" for f in vp.plan.filters):
            vp = apply_retrieval(llm, vp, parquet).plan
        result = execute_plan(vp, parquet, dataset_id=ds.id, version=ds.version)
        chart = choose_chart(vp, result)
        base.summary = fallback_summary(vp, result)
        base.columns = result.columns
        base.rows = result.rows[:TILE_ROWS]
        base.chart = chart.__dict__ if chart else None
    except Exception as exc:  # a broken tile shouldn't break the whole page
        base.error = str(exc)[:200]
        base.summary = describe_plan_safe(tile)
    return base


def describe_plan_safe(tile: dict[str, Any]) -> str:
    try:
        return describe_plan(
            validate_plan(QueryPlan.model_validate(tile["plan"]), [], date.today())
        )
    except Exception:
        return "This question couldn't be answered with the current data."


@router.get("", response_model=list[DashboardSummary])
def list_dashboards(user: CurrentUser, db: DbSession) -> list[DashboardSummary]:
    stmt = (
        select(Dashboard)
        .where(Dashboard.user_id == user.id, Dashboard.workspace_id == user.workspace_id)
        .order_by(Dashboard.created_at.desc())
    )
    return [DashboardSummary(id=d.id, name=d.name, tiles=len(d.tiles)) for d in db.scalars(stmt)]


@router.post("", response_model=DashboardSummary, status_code=status.HTTP_201_CREATED)
def create_dashboard(body: DashboardCreate, user: CurrentUser, db: DbSession) -> DashboardSummary:
    d = Dashboard(workspace_id=user.workspace_id, user_id=user.id, name=body.name, tiles=[])
    db.add(d)
    db.commit()
    return DashboardSummary(id=d.id, name=d.name, tiles=0)


@router.post("/pin", response_model=DashboardSummary)
def pin_answer(body: PinIn, user: CurrentUser, db: DbSession) -> DashboardSummary:
    """Pin a computed answer to a dashboard (an existing one, or a new one by name)."""
    reply = db.get(Message, body.message_id)
    if reply is None or reply.role != "assistant" or not reply.plan:
        raise not_found("Answer")
    conv = db.get(Conversation, reply.conversation_id)
    if conv is None or conv.workspace_id != user.workspace_id or conv.user_id != user.id:
        raise not_found("Answer")
    if (reply.plan or {}).get("metric") == "describe":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File overviews can't be pinned.")
    question = next((m for m in conv.messages if m.position == reply.position - 1), None)
    tile = {
        "dataset_id": reply.dataset_id or conv.dataset_id,
        "question": question.content[:300] if question else "Answer",
        "plan": reply.plan,
    }
    if body.dashboard_id:
        d = _own(db, user, body.dashboard_id)
    else:
        d = Dashboard(workspace_id=user.workspace_id, user_id=user.id,
                      name=(body.name or str(tile["question"]))[:200], tiles=[])  # fmt: skip
        db.add(d)
    if len(d.tiles) >= MAX_TILES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"A dashboard holds {MAX_TILES} tiles.")
    d.tiles = [*d.tiles, tile]
    db.commit()
    return DashboardSummary(id=d.id, name=d.name, tiles=len(d.tiles))


@router.get("/{dashboard_id}", response_model=DashboardView)
def view_dashboard(
    dashboard_id: str, user: CurrentUser, db: DbSession, settings: AppSettings, today: Today,
    llm: Annotated[LLMProvider, Depends(get_llm)],
) -> DashboardView:  # fmt: skip
    """Run every tile on the current data and return the results."""
    d = _own(db, user, dashboard_id)
    tiles = [_render_tile(settings, db, t, llm, today) for t in d.tiles[:MAX_TILES]]
    return DashboardView(id=d.id, name=d.name, tiles=tiles)


@router.delete("/{dashboard_id}/tiles/{index}", response_model=DashboardSummary)
def remove_tile(
    dashboard_id: str, index: int, user: CurrentUser, db: DbSession
) -> DashboardSummary:
    d = _own(db, user, dashboard_id)
    if not 0 <= index < len(d.tiles):
        raise not_found("Tile")
    d.tiles = [t for i, t in enumerate(d.tiles) if i != index]
    db.commit()
    return DashboardSummary(id=d.id, name=d.name, tiles=len(d.tiles))


@router.delete("/{dashboard_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dashboard(dashboard_id: str, user: CurrentUser, db: DbSession) -> None:
    d = _own(db, user, dashboard_id)
    db.delete(d)
    db.commit()
