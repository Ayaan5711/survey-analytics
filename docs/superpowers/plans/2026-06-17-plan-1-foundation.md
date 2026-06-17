# Survey Analytics — Plan 1: Foundation & Core Backend

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the backend for upload → profiling → auto-dashboard, with full persistence (SQLite + Parquet) and a working FastAPI server that the chat agent (Plan 2) and frontend (Plan 4) will plug into.

**Architecture:** A user uploads a CSV/Excel file; the backend parses it, computes a DatasetProfile (schema, stats, data-quality flags) cached as JSON + Parquet on disk, then generates an auto-dashboard (deterministic matplotlib charts + one Azure OpenAI narrative call). Sessions, messages, pinned charts, insights, and comparisons are stored in SQLite via SQLAlchemy. The LLM client wraps Azure OpenAI with primary/fallback model routing. This plan does NOT include the chat agent or frontend — it ends with a `/sessions/{id}/dashboard` endpoint returning JSON a future plan's UI can consume.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2 (SQLite), pandas 2, pyarrow (Parquet), matplotlib, openpyxl, rapidfuzz (fuzzy category matching), openai SDK ≥ 1.0 (Azure), pytest, httpx.

---

## File Map

```text
backend/
  pyproject.toml              — deps + project metadata
  .env.example                — required env vars with placeholders
  app/
    __init__.py
    main.py                   — FastAPI app, router registration, lifespan
    config.py                 — env var loading (Settings dataclass)
    db/
      __init__.py
      database.py             — engine, SessionLocal, Base, get_db dependency
      models.py               — ORM models: Session, ChatMessage, PinnedChart, Insight, Comparison
    data/
      __init__.py
      loader.py               — CSV/Excel → (df, schema); type detection; multi-sheet Excel
      profiler.py             — DatasetProfile dataclass + build_profile(); Parquet save/load
      quality.py              — DataQualityFlags: duplicate rows, fuzzy categories, empty cols
    llm/
      __init__.py
      client.py               — AzureOpenAIClient: async chat_completion() w/ primary/fallback
      prompts.py              — dashboard_narrative_prompt(profile) → str
    dashboard/
      __init__.py
      charts.py               — deterministic_charts(profile) → list[ChartResult]
      generator.py            — generate_dashboard(session, db) → DashboardResponse
    api/
      __init__.py
      upload.py               — POST /upload (CSV/Excel → session), POST /upload/excel-sheet
      sessions.py             — GET /sessions, GET /sessions/{id}, DELETE /sessions/{id}
      dashboard.py            — GET /sessions/{id}/dashboard
  tests/
    conftest.py               — fixtures: tmp_db, sample_csv_bytes, sample_excel_bytes, client
    test_loader.py
    test_profiler.py
    test_quality.py
    test_charts.py
    test_dashboard_generator.py
    test_api_upload.py
    test_api_sessions.py
    test_api_dashboard.py
```

---

## Task 1: Repo Scaffold

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/.env.example`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`

- [ ] **Step 1: Create `backend/pyproject.toml`**

```toml
[project]
name = "survey-analytics"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pandas>=2.2",
    "openpyxl>=3.1",
    "pyarrow>=15.0",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "Pillow>=10.0",
    "sqlalchemy>=2.0",
    "python-dotenv>=1.0",
    "openai>=1.30",
    "rapidfuzz>=3.9",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `backend/.env.example`**

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_PRIMARY=gpt-4-1
AZURE_OPENAI_DEPLOYMENT_FALLBACK=gpt-4o-mini
DATA_DIR=./data/sessions
DATABASE_URL=sqlite:///./survey_analytics.db
```

Copy to `.env` and fill in values before running.

- [ ] **Step 3: Create `backend/app/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    azure_openai_endpoint: str = field(default_factory=lambda: os.environ["AZURE_OPENAI_ENDPOINT"])
    azure_openai_api_key: str = field(default_factory=lambda: os.environ["AZURE_OPENAI_API_KEY"])
    azure_openai_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))
    deployment_primary: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_PRIMARY", "gpt-4-1"))
    deployment_fallback: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK", "gpt-4o-mini"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data/sessions")))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./survey_analytics.db"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Create `backend/app/main.py`**

```python
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.database import create_tables
from app.api import upload, sessions, dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(title="Survey Analytics", lifespan=lifespan)
app.include_router(upload.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
```

- [ ] **Step 5: Create empty `__init__.py` files**

```bash
touch backend/app/__init__.py \
      backend/app/db/__init__.py \
      backend/app/data/__init__.py \
      backend/app/llm/__init__.py \
      backend/app/dashboard/__init__.py \
      backend/app/api/__init__.py \
      backend/tests/__init__.py
```

- [ ] **Step 6: Install deps and verify FastAPI starts**

```bash
cd backend
pip install -e ".[dev]"
uvicorn app.main:app --reload
# Expected: Uvicorn running on http://127.0.0.1:8000
# GET http://127.0.0.1:8000/docs should return 200
```

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "feat: repo scaffold — FastAPI app, config, pyproject"
```

---

## Task 2: Database Models

**Files:**
- Create: `backend/app/db/database.py`
- Create: `backend/app/db/models.py`
- Test: `backend/tests/test_db.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.db.models import Base, Session as SessionModel
from app.db.database import get_db
import pytest


@pytest.fixture
def db_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return engine


def test_session_model_create(db_engine):
    with Session(db_engine) as db:
        s = SessionModel(filename="test.csv", row_count=100,
                         profile_path="/tmp/p.json", data_path="/tmp/d.parquet")
        db.add(s)
        db.commit()
        db.refresh(s)
    assert s.id is not None
    assert s.filename == "test.csv"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd backend && pytest tests/test_db.py -v
# Expected: FAIL — cannot import Session model
```

- [ ] **Step 3: Create `backend/app/db/models.py`**

```python
from __future__ import annotations
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    row_count = Column(Integer)
    profile_path = Column(String)
    data_path = Column(String)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    pinned_charts = relationship("PinnedChart", back_populates="session", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    generated_code = Column(Text, nullable=True)
    chart_paths = Column(JSON, nullable=True)
    follow_ups = Column(JSON, nullable=True)
    caveats = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="messages")


class PinnedChart(Base):
    __tablename__ = "pinned_charts"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    chart_path = Column(String, nullable=False)
    title = Column(String, nullable=True)
    source_message_id = Column(String, ForeignKey("chat_messages.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="pinned_charts")


class Insight(Base):
    __tablename__ = "insights"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    supporting_tool_calls = Column(JSON, nullable=True)
    chart_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="insights")


class Comparison(Base):
    __tablename__ = "comparisons"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    base_session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    compare_session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    diff_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 4: Create `backend/app/db/database.py`**

```python
from __future__ import annotations
from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from app.config import settings
from app.db.models import Base

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def create_tables() -> None:
    Base.metadata.create_all(engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 5: Run test to confirm pass**

```bash
pytest tests/test_db.py -v
# Expected: PASS
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/db/ backend/tests/test_db.py
git commit -m "feat: SQLAlchemy models — sessions, chat_messages, pinned_charts, insights, comparisons"
```

---

## Task 3: Data Loader

**Files:**
- Create: `backend/app/data/loader.py`
- Test: `backend/tests/test_loader.py`

Adapted from `pulseiq-mvp/app/utils/csv_loader.py` — handles messy headers, type detection, Excel multi-sheet.

- [ ] **Step 1: Create `backend/tests/conftest.py`**

```python
import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.main import app
from app.db.database import get_db
from app.db.models import Base


@pytest.fixture
def sample_csv_bytes() -> bytes:
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "Department": ["HR", "HR", "Engineering", "engineering", "Sales"],
        "Satisfaction": [4, 2, 5, 3, 4],
        "Comments": ["Great place", "Too many meetings", "", "Good team", "Loves it"],
        "Salary": [50000, 60000, 80000, 75000, 55000],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def sample_excel_bytes() -> bytes:
    df = pd.DataFrame({
        "Department": ["HR", "Engineering", "Sales"],
        "Headcount": [10, 25, 15],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Q1")
    return buf.getvalue()


@pytest.fixture
def tmp_db(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()


@pytest.fixture
def client(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")

    def override_get_db():
        yield tmp_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Write failing tests for loader**

```python
# tests/test_loader.py
import io
import pytest
import pandas as pd
from app.data.loader import load_file, detect_column_type, get_excel_sheet_names


def test_load_csv_basic(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert len(df) == 5
    assert "Department" in schema
    assert "Satisfaction" in schema


def test_load_csv_cleans_headers(sample_csv_bytes):
    """Column names should be stripped and normalised."""
    raw = b"  Name  , Score \n Alice, 5\n"
    df, schema = load_file(raw, "messy.csv")
    assert "Name" in df.columns
    assert "Score" in df.columns


def test_detect_numeric_column(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert schema["Satisfaction"]["type"] == "numeric"
    assert schema["Salary"]["type"] == "numeric"


def test_detect_categorical_column(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert schema["Department"]["type"] == "categorical"


def test_detect_open_text_column(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert schema["Comments"]["type"] == "open_text"


def test_get_excel_sheet_names(sample_excel_bytes):
    sheets = get_excel_sheet_names(sample_excel_bytes)
    assert sheets == ["Q1"]


def test_load_excel_single_sheet(sample_excel_bytes):
    df, schema = load_file(sample_excel_bytes, "data.xlsx", sheet_name="Q1")
    assert "Department" in df.columns
    assert len(df) == 3
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/test_loader.py -v
# Expected: FAIL — cannot import load_file
```

- [ ] **Step 4: Create `backend/app/data/loader.py`**

```python
from __future__ import annotations
import io
import re
from typing import Any
import numpy as np
import pandas as pd


_OPEN_TEXT_MIN_AVG_LEN = 20
_OPEN_TEXT_MIN_UNIQUE_RATIO = 0.5
_CATEGORICAL_MAX_UNIQUE_RATIO = 0.2
_CATEGORICAL_MAX_UNIQUE_ABS = 50


def detect_column_type(series: pd.Series) -> str:
    """Classify a column as numeric / categorical / open_text / datetime / unknown."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return "unknown"
    avg_len = non_null.str.len().mean()
    unique_ratio = non_null.nunique() / len(non_null)
    if avg_len >= _OPEN_TEXT_MIN_AVG_LEN and unique_ratio >= _OPEN_TEXT_MIN_UNIQUE_RATIO:
        return "open_text"
    if unique_ratio <= _CATEGORICAL_MAX_UNIQUE_RATIO or non_null.nunique() <= _CATEGORICAL_MAX_UNIQUE_ABS:
        return "categorical"
    return "open_text"


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    return df


def get_excel_sheet_names(content: bytes) -> list[str]:
    xl = pd.ExcelFile(io.BytesIO(content))
    return xl.sheet_names


def load_file(
    content: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Parse CSV or Excel bytes → (DataFrame, schema).

    schema shape: {col_name: {"type": str, "n_unique": int}}
    Raises ValueError for empty files or unsupported formats.
    """
    if not content:
        raise ValueError("File is empty")

    lower = filename.lower()
    if lower.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Could not parse CSV: {exc}") from exc
    elif lower.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name or 0)
        except Exception as exc:
            raise ValueError(f"Could not parse Excel: {exc}") from exc
    else:
        raise ValueError(f"Unsupported file type: {filename}. Expected .csv, .xlsx, or .xls")

    if df.empty:
        raise ValueError("File contains no data rows")

    df = _clean_headers(df)

    schema: dict[str, dict[str, Any]] = {
        col: {
            "type": detect_column_type(df[col]),
            "n_unique": int(df[col].nunique()),
        }
        for col in df.columns
    }
    return df, schema
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
pytest tests/test_loader.py -v
# Expected: all PASS
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/data/loader.py backend/tests/test_loader.py backend/tests/conftest.py
git commit -m "feat: data loader — CSV/Excel parsing, type detection, Excel sheet picker"
```

---

## Task 4: Dataset Profiler

**Files:**
- Create: `backend/app/data/profiler.py`
- Test: `backend/tests/test_profiler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_profiler.py
import json
import pytest
from app.data.loader import load_file
from app.data.profiler import build_profile, load_profile, DatasetProfile


def test_build_profile_shape(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(
        df=df, schema=schema, session_id="sess-1",
        filename="responses.csv", data_dir=tmp_path
    )
    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 5
    assert profile.col_count == 5
    assert "Satisfaction" in profile.columns
    assert "Department" in profile.columns


def test_numeric_column_has_stats(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    col = profile.columns["Satisfaction"]
    assert col.dtype == "numeric"
    assert col.mean is not None
    assert col.min is not None


def test_categorical_column_has_top_values(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    col = profile.columns["Department"]
    assert col.dtype == "categorical"
    assert isinstance(col.top_values, dict)
    assert len(col.top_values) > 0


def test_profile_persisted_as_parquet_and_json(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    assert (tmp_path / "sess-1" / "data.parquet").exists()
    assert (tmp_path / "sess-1" / "profile.json").exists()


def test_load_profile_round_trip(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    original = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    loaded = load_profile(tmp_path / "sess-1" / "profile.json")
    assert loaded.row_count == original.row_count
    assert set(loaded.columns.keys()) == set(original.columns.keys())


def test_sample_rows_count(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    assert len(profile.sample_rows) <= 20
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_profiler.py -v
# Expected: FAIL — cannot import build_profile
```

- [ ] **Step 3: Create `backend/app/data/profiler.py`**

```python
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    dtype: str
    missing_pct: float
    n_unique: int
    # numeric
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    # categorical
    top_values: dict[str, int] = field(default_factory=dict)
    # all
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetProfile:
    session_id: str
    filename: str
    row_count: int
    col_count: int
    columns: dict[str, ColumnProfile]
    sample_rows: list[dict]
    open_text_columns: list[str]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _profile_column(series: pd.Series, dtype: str) -> ColumnProfile:
    total = len(series)
    missing_pct = round(series.isna().sum() / total * 100, 2) if total else 0.0
    n_unique = int(series.nunique())
    non_null = series.dropna()
    sample = [v for v in non_null.head(5).tolist() if v is not None]

    cp = ColumnProfile(
        dtype=dtype,
        missing_pct=missing_pct,
        n_unique=n_unique,
        sample_values=sample,
    )

    if dtype == "numeric":
        cp.min = float(non_null.min()) if len(non_null) else None
        cp.max = float(non_null.max()) if len(non_null) else None
        cp.mean = round(float(non_null.mean()), 4) if len(non_null) else None
        cp.median = float(non_null.median()) if len(non_null) else None
        cp.std = round(float(non_null.std()), 4) if len(non_null) else None
    elif dtype == "categorical":
        counts = non_null.astype(str).value_counts().head(10)
        cp.top_values = {k: int(v) for k, v in counts.items()}

    return cp


def build_profile(
    df: pd.DataFrame,
    schema: dict[str, dict[str, Any]],
    session_id: str,
    filename: str,
    data_dir: Path,
) -> DatasetProfile:
    session_dir = data_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(session_dir / "data.parquet", index=False)

    columns = {
        col: _profile_column(df[col], schema[col]["type"])
        for col in df.columns
    }
    open_text_cols = [c for c, v in schema.items() if v["type"] == "open_text"]
    sample_rows = df.head(20).replace({np.nan: None}).to_dict(orient="records")

    profile = DatasetProfile(
        session_id=session_id,
        filename=filename,
        row_count=len(df),
        col_count=len(df.columns),
        columns=columns,
        sample_rows=sample_rows,
        open_text_columns=open_text_cols,
    )

    profile_path = session_dir / "profile.json"
    profile_path.write_text(json.dumps(_serialize(asdict(profile)), indent=2))
    return profile


def load_profile(path: Path) -> DatasetProfile:
    data = json.loads(path.read_text())
    columns = {k: ColumnProfile(**v) for k, v in data.pop("columns").items()}
    return DatasetProfile(**data, columns=columns)


def _serialize(obj: Any) -> Any:
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_profiler.py -v
# Expected: all PASS
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/data/profiler.py backend/tests/test_profiler.py
git commit -m "feat: dataset profiler — build_profile, ColumnProfile, Parquet+JSON persistence"
```

---

## Task 5: Data Quality Checker

**Files:**
- Create: `backend/app/data/quality.py`
- Test: `backend/tests/test_quality.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_quality.py
import io
import pytest
import pandas as pd
from app.data.quality import check_quality, DataQualityFlags


def _df(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)


def test_detects_duplicate_rows():
    df = _df({"A": [1, 1, 2], "B": ["x", "x", "y"]})
    flags = check_quality(df, {"A": {"type": "numeric"}, "B": {"type": "categorical"}})
    assert flags.duplicate_rows == 1


def test_no_duplicates():
    df = _df({"A": [1, 2, 3]})
    flags = check_quality(df, {"A": {"type": "numeric"}})
    assert flags.duplicate_rows == 0


def test_detects_fuzzy_category_issues():
    df = _df({"Dept": ["HR", "hr", "HR", "Engineering", "engineering"]})
    flags = check_quality(df, {"Dept": {"type": "categorical"}})
    issue_cols = [i["column"] for i in flags.fuzzy_category_issues]
    assert "Dept" in issue_cols


def test_detects_mostly_empty_columns():
    df = _df({"A": [1, None, None, None, None, None, None, None, None, None, None]})
    flags = check_quality(df, {"A": {"type": "numeric"}})
    assert "A" in flags.mostly_empty_columns


def test_detects_constant_columns():
    df = _df({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]})
    flags = check_quality(df, {"A": {"type": "numeric"}, "B": {"type": "numeric"}})
    assert "A" in flags.constant_columns
    assert "B" not in flags.constant_columns
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_quality.py -v
# Expected: FAIL
```

- [ ] **Step 3: Create `backend/app/data/quality.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from rapidfuzz import fuzz


_FUZZY_THRESHOLD = 85
_MOSTLY_EMPTY_PCT = 90.0


@dataclass
class DataQualityFlags:
    duplicate_rows: int = 0
    fuzzy_category_issues: list[dict] = field(default_factory=list)
    mostly_empty_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)


def _find_fuzzy_pairs(values: list[str]) -> list[tuple[str, str]]:
    """Return pairs of values that look like the same thing with different casing/spacing."""
    pairs: list[tuple[str, str]] = []
    seen: set[int] = set()
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            if j <= i or (i, j) in seen:
                continue
            if a.lower() == b.lower():
                pairs.append((a, b))
                seen.add((i, j))
            elif fuzz.ratio(a.lower(), b.lower()) >= _FUZZY_THRESHOLD:
                pairs.append((a, b))
                seen.add((i, j))
    return pairs


def check_quality(df: pd.DataFrame, schema: dict[str, dict[str, Any]]) -> DataQualityFlags:
    flags = DataQualityFlags()

    flags.duplicate_rows = int(df.duplicated().sum())

    total = len(df)
    for col, meta in schema.items():
        if col not in df.columns:
            continue
        series = df[col]
        missing_pct = series.isna().sum() / total * 100 if total else 0
        if missing_pct >= _MOSTLY_EMPTY_PCT:
            flags.mostly_empty_columns.append(col)
        if series.nunique() == 1:
            flags.constant_columns.append(col)
        if meta["type"] == "categorical":
            unique_vals = [str(v) for v in series.dropna().unique().tolist()]
            pairs = _find_fuzzy_pairs(unique_vals)
            if pairs:
                flags.fuzzy_category_issues.append({
                    "column": col,
                    "pairs": [{"a": a, "b": b} for a, b in pairs],
                    "suggestion": f"Found {len(pairs)} near-duplicate value(s) — consider standardising."
                })

    return flags
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_quality.py -v
# Expected: all PASS
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/data/quality.py backend/tests/test_quality.py
git commit -m "feat: data quality checker — duplicates, fuzzy categories, empty/constant columns"
```

---

## Task 6: Azure OpenAI Client

**Files:**
- Create: `backend/app/llm/client.py`
- Create: `backend/app/llm/prompts.py`
- Test: `backend/tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_client.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.llm.client import AzureOpenAIClient


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    return AzureOpenAIClient()


@pytest.mark.asyncio
async def test_chat_completion_returns_string(mock_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Analysis result"

    with patch.object(mock_client._client.chat.completions, "create",
                      new=AsyncMock(return_value=mock_response)):
        result = await mock_client.chat_completion(
            messages=[{"role": "user", "content": "Summarise this data"}],
            use_fallback=False,
        )
    assert result == "Analysis result"


@pytest.mark.asyncio
async def test_chat_completion_uses_fallback_model(mock_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    calls = []
    async def fake_create(**kwargs):
        calls.append(kwargs.get("model"))
        return mock_response

    with patch.object(mock_client._client.chat.completions, "create", side_effect=fake_create):
        await mock_client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            use_fallback=True,
        )
    assert calls[0] == mock_client.deployment_fallback
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_llm_client.py -v
# Expected: FAIL
```

- [ ] **Step 3: Create `backend/app/llm/client.py`**

```python
from __future__ import annotations
from openai import AsyncAzureOpenAI
from app.config import settings


class AzureOpenAIClient:
    def __init__(self) -> None:
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.deployment_primary = settings.deployment_primary
        self.deployment_fallback = settings.deployment_fallback

    async def chat_completion(
        self,
        messages: list[dict],
        use_fallback: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str:
        model = self.deployment_fallback if use_fallback else self.deployment_primary
        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response_format:
            kwargs["response_format"] = response_format
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# module-level singleton — imported by consumers
llm = AzureOpenAIClient()
```

- [ ] **Step 4: Create `backend/app/llm/prompts.py`**

```python
from __future__ import annotations
import json
from app.data.profiler import DatasetProfile


def dashboard_narrative_prompt(profile: DatasetProfile) -> str:
    col_summary = []
    for name, col in profile.columns.items():
        if col.dtype == "numeric":
            col_summary.append(
                f"- {name} (numeric): mean={col.mean}, min={col.min}, max={col.max}, "
                f"missing={col.missing_pct}%"
            )
        elif col.dtype == "categorical":
            top = list(col.top_values.items())[:3]
            col_summary.append(
                f"- {name} (categorical): top values = {top}, n_unique={col.n_unique}, "
                f"missing={col.missing_pct}%"
            )
        else:
            col_summary.append(f"- {name} ({col.dtype}): missing={col.missing_pct}%")

    return f"""You are a data analyst. Below is a summary of a survey dataset called "{profile.filename}".

Dataset: {profile.row_count} rows, {profile.col_count} columns.

Column statistics:
{chr(10).join(col_summary)}

Write a short (3-5 sentence) plain-English summary highlighting the most notable patterns,
potential issues, or interesting findings. Be specific — mention column names and numbers.
Do not use bullet points. Do not suggest next steps."""
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
pytest tests/test_llm_client.py -v
# Expected: all PASS
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/llm/ backend/tests/test_llm_client.py
git commit -m "feat: Azure OpenAI client — primary/fallback routing, dashboard narrative prompt"
```

---

## Task 7: Dashboard Charts (Deterministic)

**Files:**
- Create: `backend/app/dashboard/charts.py`
- Test: `backend/tests/test_charts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_charts.py
import pytest
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.dashboard.charts import deterministic_charts, ChartResult


def test_returns_list_of_chart_results(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    charts = deterministic_charts(profile, tmp_path / "sess-1")
    assert isinstance(charts, list)
    assert all(isinstance(c, ChartResult) for c in charts)


def test_chart_results_have_png_bytes(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    charts = deterministic_charts(profile, tmp_path / "sess-1")
    assert len(charts) >= 1
    for chart in charts:
        assert isinstance(chart.png_bytes, bytes)
        assert len(chart.png_bytes) > 0


def test_chart_has_title(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    charts = deterministic_charts(profile, tmp_path / "sess-1")
    for c in charts:
        assert c.title
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_charts.py -v
# Expected: FAIL
```

- [ ] **Step 3: Create `backend/app/dashboard/charts.py`**

```python
from __future__ import annotations
import io
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from app.data.profiler import DatasetProfile


@dataclass
class ChartResult:
    title: str
    chart_type: str       # "bar" | "histogram" | "pie"
    png_bytes: bytes
    filename: str


def _save_fig(title: str, chart_type: str, out_dir: Path, idx: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    png = buf.read()
    fname = f"chart_{idx}_{chart_type}.png"
    (out_dir / fname).write_bytes(png)
    return png, fname


def deterministic_charts(profile: DatasetProfile, out_dir: Path) -> list[ChartResult]:
    """Generate up to 3 standard charts from the profile JSON (no DataFrame needed)."""
    results: list[ChartResult] = []
    idx = 0

    # Bar chart: top categorical column by count
    cat_cols = [(c, p) for c, p in profile.columns.items() if p.dtype == "categorical" and p.top_values]
    if cat_cols:
        col_name, col_p = cat_cols[0]
        labels = list(col_p.top_values.keys())
        values = list(col_p.top_values.values())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels[:10], values[:10], color="black")
        ax.set_title(f"{col_name} — response counts", fontsize=12)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        png, fname = _save_fig(f"{col_name} counts", "bar", out_dir, idx)
        results.append(ChartResult(title=f"{col_name} — distribution", chart_type="bar",
                                   png_bytes=png, filename=fname))
        idx += 1

    # Histogram: first numeric column
    num_cols = [(c, p) for c, p in profile.columns.items() if p.dtype == "numeric"]
    if num_cols:
        col_name, col_p = num_cols[0]
        sample_vals = [v for v in col_p.sample_values if v is not None]
        if sample_vals:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(sample_vals, bins=min(10, len(sample_vals)), color="black", edgecolor="white")
            ax.set_title(f"{col_name} — distribution (sample)", fontsize=12)
            ax.set_xlabel(col_name)
            ax.set_ylabel("Frequency")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            png, fname = _save_fig(f"{col_name} histogram", "histogram", out_dir, idx)
            results.append(ChartResult(title=f"{col_name} — histogram", chart_type="histogram",
                                       png_bytes=png, filename=fname))
            idx += 1

    # Missing data bar (if any column has >0% missing)
    missing = {c: p.missing_pct for c, p in profile.columns.items() if p.missing_pct > 0}
    if missing:
        fig, ax = plt.subplots(figsize=(7, 4))
        cols = list(missing.keys())[:10]
        pcts = [missing[c] for c in cols]
        ax.barh(cols, pcts, color="black")
        ax.set_title("Missing data %", fontsize=12)
        ax.set_xlabel("Missing %")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        png, fname = _save_fig("Missing data", "bar", out_dir, idx)
        results.append(ChartResult(title="Missing data", chart_type="bar",
                                   png_bytes=png, filename=fname))

    return results
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_charts.py -v
# Expected: all PASS
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/dashboard/charts.py backend/tests/test_charts.py
git commit -m "feat: deterministic dashboard charts — bar, histogram, missing data (matplotlib)"
```

---

## Task 8: Dashboard Generator

**Files:**
- Create: `backend/app/dashboard/generator.py`
- Test: `backend/tests/test_dashboard_generator.py`

- [ ] **Step 1: Write failing tests (with mocked LLM)**

```python
# tests/test_dashboard_generator.py
import base64
import pytest
from unittest.mock import AsyncMock, patch
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.data.quality import check_quality
from app.dashboard.generator import generate_dashboard, DashboardResponse


@pytest.mark.asyncio
async def test_generate_dashboard_returns_response(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    quality = check_quality(df, schema)

    with patch("app.dashboard.generator.llm.chat_completion",
               new=AsyncMock(return_value="This dataset has 5 rows and looks interesting.")):
        result = await generate_dashboard(profile, quality, tmp_path / "sess-1")

    assert isinstance(result, DashboardResponse)
    assert result.narrative
    assert isinstance(result.charts, list)
    assert len(result.charts) >= 1


@pytest.mark.asyncio
async def test_dashboard_charts_are_base64(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    quality = check_quality(df, schema)

    with patch("app.dashboard.generator.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        result = await generate_dashboard(profile, quality, tmp_path / "sess-1")

    for chart in result.charts:
        # Should be valid base64
        base64.b64decode(chart["png_b64"])


@pytest.mark.asyncio
async def test_dashboard_includes_quality_flags(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    quality = check_quality(df, schema)

    with patch("app.dashboard.generator.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        result = await generate_dashboard(profile, quality, tmp_path / "sess-1")

    assert result.quality_flags is not None
    assert hasattr(result.quality_flags, "duplicate_rows")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_dashboard_generator.py -v
# Expected: FAIL
```

- [ ] **Step 3: Create `backend/app/dashboard/generator.py`**

```python
from __future__ import annotations
import base64
from dataclasses import asdict, dataclass
from pathlib import Path
from app.data.profiler import DatasetProfile
from app.data.quality import DataQualityFlags
from app.dashboard.charts import deterministic_charts, ChartResult
from app.llm.client import llm
from app.llm.prompts import dashboard_narrative_prompt


@dataclass
class DashboardResponse:
    session_id: str
    filename: str
    row_count: int
    col_count: int
    narrative: str
    charts: list[dict]              # [{title, chart_type, png_b64, filename}]
    quality_flags: DataQualityFlags
    open_text_columns: list[str]
    column_summary: dict            # {col_name: {dtype, missing_pct, n_unique, ...}}


async def generate_dashboard(
    profile: DatasetProfile,
    quality: DataQualityFlags,
    session_dir: Path,
) -> DashboardResponse:
    charts: list[ChartResult] = deterministic_charts(profile, session_dir)

    prompt = dashboard_narrative_prompt(profile)
    narrative = await llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        use_fallback=False,
        max_tokens=300,
        temperature=0.4,
    )

    charts_out = [
        {
            "title": c.title,
            "chart_type": c.chart_type,
            "png_b64": base64.b64encode(c.png_bytes).decode(),
            "filename": c.filename,
        }
        for c in charts
    ]

    col_summary = {
        name: {
            "dtype": cp.dtype,
            "missing_pct": cp.missing_pct,
            "n_unique": cp.n_unique,
            "mean": cp.mean,
            "top_values": dict(list(cp.top_values.items())[:5]),
        }
        for name, cp in profile.columns.items()
    }

    return DashboardResponse(
        session_id=profile.session_id,
        filename=profile.filename,
        row_count=profile.row_count,
        col_count=profile.col_count,
        narrative=narrative,
        charts=charts_out,
        quality_flags=quality,
        open_text_columns=profile.open_text_columns,
        column_summary=col_summary,
    )
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_dashboard_generator.py -v
# Expected: all PASS
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/dashboard/ backend/tests/test_dashboard_generator.py
git commit -m "feat: dashboard generator — deterministic charts + LLM narrative + quality flags"
```

---

## Task 9: Upload API Route

**Files:**
- Create: `backend/app/api/upload.py`
- Test: `backend/tests/test_api_upload.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api_upload.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_upload_csv_returns_session(client, sample_csv_bytes):
    with patch("app.api.upload.llm.chat_completion",
               new=AsyncMock(return_value="Narrative text.")):
        r = client.post("/api/upload",
                        files={"file": ("responses.csv", sample_csv_bytes, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["row_count"] == 5
    assert "columns" in data


@pytest.mark.asyncio
async def test_upload_excel_single_sheet(client, sample_excel_bytes):
    with patch("app.api.upload.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        r = client.post("/api/upload",
                        files={"file": ("data.xlsx", sample_excel_bytes,
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 200
    data = r.json()
    assert data["row_count"] == 3


def test_upload_empty_file_returns_400(client):
    r = client.post("/api/upload",
                    files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_upload_unsupported_format_returns_400(client):
    r = client.post("/api/upload",
                    files={"file": ("data.txt", b"hello", "text/plain")})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_upload_excel_multi_sheet_returns_sheets(client):
    """Multi-sheet Excel upload returns 409 with sheet list for the client to pick."""
    import io, pandas as pd
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"A": [1]}).to_excel(w, sheet_name="Sheet1", index=False)
        pd.DataFrame({"B": [2]}).to_excel(w, sheet_name="Sheet2", index=False)
    buf.seek(0)
    r = client.post("/api/upload",
                    files={"file": ("multi.xlsx", buf.read(),
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 409
    body = r.json()
    assert "sheets" in body["detail"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_api_upload.py -v
# Expected: FAIL
```

- [ ] **Step 3: Create `backend/app/api/upload.py`**

```python
from __future__ import annotations
import logging
from dataclasses import asdict
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from fastapi import Depends
from app.config import settings
from app.data.loader import get_excel_sheet_names, load_file
from app.data.profiler import build_profile
from app.data.quality import check_quality
from app.dashboard.generator import generate_dashboard
from app.db.database import get_db
from app.db.models import Session as SessionModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    lower = (file.filename or "").lower()
    if not (lower.endswith(".csv") or lower.endswith(".xlsx") or lower.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Unsupported file type — upload .csv, .xlsx, or .xls")

    # Multi-sheet Excel gate
    if lower.endswith((".xlsx", ".xls")):
        sheets = get_excel_sheet_names(content)
        if len(sheets) > 1:
            raise HTTPException(
                status_code=409,
                detail={"message": "Excel file has multiple sheets — pick one", "sheets": sheets},
            )

    try:
        df, schema = load_file(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = SessionModel(filename=file.filename, row_count=len(df))
    db.add(record)
    db.commit()
    db.refresh(record)

    try:
        profile = build_profile(df, schema, record.id, file.filename, settings.data_dir)
        quality = check_quality(df, schema)
    except Exception as exc:
        logger.error(f"Profiling failed: {exc}")
        raise HTTPException(status_code=500, detail="Failed to profile data")

    session_dir = settings.data_dir / record.id
    record.profile_path = str(session_dir / "profile.json")
    record.data_path = str(session_dir / "data.parquet")
    db.commit()

    dashboard = await generate_dashboard(profile, quality, session_dir)

    return {
        "session_id": record.id,
        "filename": file.filename,
        "row_count": profile.row_count,
        "col_count": profile.col_count,
        "columns": list(profile.columns.keys()),
        "open_text_columns": profile.open_text_columns,
        "dashboard": {
            "narrative": dashboard.narrative,
            "charts": dashboard.charts,
            "quality_flags": asdict(dashboard.quality_flags),
            "column_summary": dashboard.column_summary,
        },
    }


@router.post("/upload/sheet")
async def upload_excel_with_sheet(
    file: UploadFile = File(...),
    sheet_name: str = "",
    db: Session = Depends(get_db),
) -> dict:
    """Re-upload the same Excel bytes with a chosen sheet_name (client called after 409)."""
    if not sheet_name:
        raise HTTPException(status_code=400, detail="sheet_name is required")
    content = await file.read()
    try:
        df, schema = load_file(content, file.filename or "upload.xlsx", sheet_name=sheet_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = SessionModel(filename=f"{file.filename} [{sheet_name}]", row_count=len(df))
    db.add(record)
    db.commit()
    db.refresh(record)

    profile = build_profile(df, schema, record.id, record.filename, settings.data_dir)
    quality = check_quality(df, schema)
    session_dir = settings.data_dir / record.id
    record.profile_path = str(session_dir / "profile.json")
    record.data_path = str(session_dir / "data.parquet")
    db.commit()

    dashboard = await generate_dashboard(profile, quality, session_dir)
    return {
        "session_id": record.id,
        "filename": record.filename,
        "row_count": profile.row_count,
        "col_count": profile.col_count,
        "columns": list(profile.columns.keys()),
        "open_text_columns": profile.open_text_columns,
        "dashboard": {
            "narrative": dashboard.narrative,
            "charts": dashboard.charts,
            "quality_flags": asdict(dashboard.quality_flags),
            "column_summary": dashboard.column_summary,
        },
    }
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_api_upload.py -v
# Expected: all PASS
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/upload.py backend/tests/test_api_upload.py
git commit -m "feat: upload API — CSV/Excel upload, profile, auto-dashboard, multi-sheet gate"
```

---

## Task 10: Sessions & Dashboard API Routes

**Files:**
- Create: `backend/app/api/sessions.py`
- Create: `backend/app/api/dashboard.py`
- Test: `backend/tests/test_api_sessions.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api_sessions.py
import pytest
from unittest.mock import AsyncMock, patch


def _upload(client, csv_bytes):
    with patch("app.api.upload.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        r = client.post("/api/upload",
                        files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_list_sessions_empty(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert r.json() == []


def test_list_sessions_after_upload(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.get("/api/sessions")
    assert r.status_code == 200
    ids = [s["session_id"] for s in r.json()]
    assert sid in ids


def test_get_session_by_id(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["session_id"] == sid


def test_get_session_not_found(client):
    r = client.get("/api/sessions/nonexistent")
    assert r.status_code == 404


def test_delete_session(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200
    r2 = client.get(f"/api/sessions/{sid}")
    assert r2.status_code == 404


def test_get_dashboard_returns_narrative(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.get(f"/api/sessions/{sid}/dashboard")
    assert r.status_code == 200
    data = r.json()
    assert "narrative" in data
    assert "charts" in data
    assert "quality_flags" in data
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_api_sessions.py -v
# Expected: FAIL
```

- [ ] **Step 3: Create `backend/app/api/sessions.py`**

```python
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel

router = APIRouter()


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.query(SessionModel).order_by(SessionModel.uploaded_at.desc()).all()
    return [
        {
            "session_id": s.id,
            "filename": s.filename,
            "row_count": s.row_count,
            "uploaded_at": s.uploaded_at.isoformat() if s.uploaded_at else None,
        }
        for s in rows
    ]


@router.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    s = db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": s.id,
        "filename": s.filename,
        "row_count": s.row_count,
        "uploaded_at": s.uploaded_at.isoformat() if s.uploaded_at else None,
        "profile_path": s.profile_path,
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    s = db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(s)
    db.commit()
    return {"deleted": session_id}
```

- [ ] **Step 4: Create `backend/app/api/dashboard.py`**

```python
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel
from app.data.profiler import load_profile
from app.data.quality import check_quality, DataQualityFlags
import pandas as pd
import json

router = APIRouter()


@router.get("/sessions/{session_id}/dashboard")
async def get_dashboard(session_id: str, db: Session = Depends(get_db)) -> dict:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    if not record.profile_path or not Path(record.profile_path).exists():
        raise HTTPException(status_code=404, detail="Profile not found — re-upload the file")

    profile = load_profile(Path(record.profile_path))

    # Recompute quality from Parquet (cheap, already on disk)
    df = pd.read_parquet(record.data_path)
    schema = {col: {"type": cp.dtype, "n_unique": cp.n_unique}
              for col, cp in profile.columns.items()}
    quality = check_quality(df, schema)

    session_dir = Path(record.profile_path).parent

    # Re-use cached charts if they exist; otherwise regenerate without LLM call
    from app.dashboard.charts import deterministic_charts
    import base64
    charts = deterministic_charts(profile, session_dir)
    charts_out = [
        {
            "title": c.title,
            "chart_type": c.chart_type,
            "png_b64": base64.b64encode(c.png_bytes).decode(),
            "filename": c.filename,
        }
        for c in charts
    ]

    # Narrative: read from disk if already generated (avoids re-calling LLM on every GET)
    narrative_path = session_dir / "narrative.txt"
    if narrative_path.exists():
        narrative = narrative_path.read_text()
    else:
        from app.llm.prompts import dashboard_narrative_prompt
        from app.llm.client import llm
        prompt = dashboard_narrative_prompt(profile)
        narrative = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            use_fallback=False,
            max_tokens=300,
            temperature=0.4,
        )
        narrative_path.write_text(narrative)

    return {
        "session_id": session_id,
        "filename": profile.filename,
        "row_count": profile.row_count,
        "col_count": profile.col_count,
        "narrative": narrative,
        "charts": charts_out,
        "quality_flags": asdict(quality),
        "column_summary": {
            name: {
                "dtype": cp.dtype,
                "missing_pct": cp.missing_pct,
                "n_unique": cp.n_unique,
                "mean": cp.mean,
                "top_values": dict(list(cp.top_values.items())[:5]),
            }
            for name, cp in profile.columns.items()
        },
        "open_text_columns": profile.open_text_columns,
    }
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
pytest tests/test_api_sessions.py -v
# Expected: all PASS
```

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
# Expected: all tests PASS
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/sessions.py backend/app/api/dashboard.py backend/tests/test_api_sessions.py
git commit -m "feat: sessions + dashboard API routes — list, get, delete, GET /dashboard with narrative cache"
```

---

## Task 11: Smoke Test End-to-End

**Goal:** Verify the full upload → profile → auto-dashboard flow works against a real running server (no mocks).

> **Note:** This task requires a real `.env` file with valid Azure OpenAI credentials.

- [ ] **Step 1: Start the server**

```bash
cd backend
uvicorn app.main:app --reload
```

- [ ] **Step 2: Upload the sample file**

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@../pulseiq-mvp/tests/fixtures/sample_survey.csv" | python -m json.tool
```

Expected: JSON with `session_id`, `row_count`, `dashboard.narrative` (a real sentence from GPT), and `dashboard.charts` (list with `png_b64` entries).

- [ ] **Step 3: List sessions**

```bash
curl http://localhost:8000/api/sessions | python -m json.tool
# Expected: list with one entry matching the uploaded file
```

- [ ] **Step 4: Get the dashboard**

```bash
SESSION_ID=<id from step 2>
curl http://localhost:8000/api/sessions/$SESSION_ID/dashboard | python -m json.tool
# Expected: same dashboard JSON, narrative served from cache (no LLM call this time)
```

- [ ] **Step 5: Commit**

```bash
git commit --allow-empty -m "chore: Plan 1 complete — foundation + core backend verified end-to-end"
```

---

## Summary

Plan 1 delivers a fully working backend:
- `POST /api/upload` — parses any CSV/Excel, profiles it, stores session, returns auto-dashboard
- `GET /api/sessions` / `GET /api/sessions/{id}` / `DELETE /api/sessions/{id}` — session management
- `GET /api/sessions/{id}/dashboard` — cached dashboard (deterministic charts + LLM narrative)
- Full test coverage for loader, profiler, quality checker, chart generation, and all routes

**Plan 2** (Chat Agent) will add `POST /api/sessions/{id}/chat` with Tier-1 tools and Tier-2 sandboxed code-gen.
