"""Liveness/readiness endpoint."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.db import get_db

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    env: str
    database: Literal["ok", "error"]
    llm_provider: str
    llm_model: str


@router.get("/health", response_model=HealthResponse)
def health(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        db_status: Literal["ok", "error"] = "ok"
    except SQLAlchemyError:
        db_status = "error"

    if db_status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        app=settings.app_name,
        env=settings.app_env,
        database=db_status,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )
