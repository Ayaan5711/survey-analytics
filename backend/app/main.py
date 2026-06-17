from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.db.database import create_tables
from app.api import upload, sessions, dashboard, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    create_tables()
    yield


app = FastAPI(title="Survey Analytics", lifespan=lifespan)
app.include_router(upload.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
