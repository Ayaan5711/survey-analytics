from __future__ import annotations
from collections.abc import Generator
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from app.config import settings
from app.db.models import Base

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# Lightweight additive migrations: columns added after a DB was first created.
# (table, column, SQLite column definition)
_ADDITIVE_COLUMNS = [
    ("chat_messages", "data_table", "JSON"),
]


def _apply_additive_migrations() -> None:
    """Add new nullable columns to existing tables so older DBs keep working."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column, coltype in _ADDITIVE_COLUMNS:
            if table not in existing_tables:
                continue
            cols = {c["name"] for c in inspector.get_columns(table)}
            if column not in cols:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {coltype}'))


def create_tables() -> None:
    Base.metadata.create_all(engine)
    _apply_additive_migrations()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
