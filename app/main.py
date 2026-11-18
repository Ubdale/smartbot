"""FastAPI application factory."""

import logging

from fastapi import FastAPI

from app.api import auth, health, workspaces
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs" if settings.app_env != "prod" else None,
    )
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(workspaces.router)
    return app


app = create_app()
