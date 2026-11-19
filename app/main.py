"""FastAPI application factory."""

import logging

from fastapi import FastAPI

from app.api import auth, chat, datasets, digests, health, team, workspaces
from app.core.config import get_settings
from app.core.headers import SecurityHeadersMiddleware
from app.core.limits import BodySizeLimitMiddleware

MULTIPART_OVERHEAD = 64 * 1024  # form fields + boundaries around the file


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs" if settings.app_env != "prod" else None,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_upload_mb * 1024 * 1024 + MULTIPART_OVERHEAD,
        path_prefixes=("/datasets",),
    )
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.app_env == "prod")
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(team.router)  # before /workspaces/{id} so "me/..." routes match
    app.include_router(workspaces.router)
    app.include_router(datasets.router)
    app.include_router(chat.router)
    app.include_router(digests.router)
    return app


app = create_app()
