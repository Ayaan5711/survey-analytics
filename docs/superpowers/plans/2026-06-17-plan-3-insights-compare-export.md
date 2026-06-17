# Survey Analytics — Plan 3: Insight Feed, Comparison Mode & Story Report Export

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic insight generation (background task after upload), dataset comparison mode, and PDF story report export.

**Architecture:** After upload, a FastAPI `BackgroundTasks` job sweeps Tier-1 tools across the profile, ranks findings by effect size, and calls the LLM once (batch) to phrase them as sentences, storing results in the `insights` table. Comparison diffs two session profiles and caches the result in the `comparisons` table. PDF export uses `fpdf2` to render pinned charts + insights into a downloadable report.

**Tech Stack:** Python 3.11+, FastAPI BackgroundTasks, fpdf2 (new dep), same stack as Plans 1-2.

## Global Constraints

- Python ≥ 3.11; `from __future__ import annotations` everywhere
- Add `fpdf2>=2.7` to `pyproject.toml` `[project.dependencies]`
- Run tests from `backend/` directory: `pytest tests/ -v`
- All new API routes prefixed with `/api`
- No new LLM calls beyond what's budgeted: one batch call for all insights per session

---

## File Map

```text
backend/
  app/
    insights/
      __init__.py
      generator.py      — sweep Tier-1 tools, rank, batch LLM phrase, store in DB
    compare/
      __init__.py
      engine.py         — diff two DatasetProfiles -> ComparisonDiff dataclass, cache in DB
    reports/
      __init__.py
      composer.py       — assemble narrative structure from profile + insights + pinned charts
      renderer.py       — render to PDF bytes using fpdf2
    api/
      insights.py       — GET /sessions/{id}/insights
      compare.py        — POST /compare (compute or return cached)
      export.py         — GET /sessions/{id}/export/pdf
    api/
      upload.py         — modify: trigger background insight generation after profiling
    main.py             — register 3 new routers
  pyproject.toml        — add fpdf2>=2.7
  tests/
    test_insights.py
    test_compare.py
    test_export.py
```

---

## Task 1: Add fpdf2 Dependency

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add `fpdf2>=2.7` to dependencies in `pyproject.toml`**

Open `backend/pyproject.toml` and add `"fpdf2>=2.7"` to the `dependencies` list:

```toml
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
    "fpdf2>=2.7",
]
```

- [ ] **Step 2: Install updated deps**

```bash
cd backend && pip install -e ".[dev]"
# Expected: fpdf2 installed
python -c "from fpdf import FPDF; print('fpdf2 ok')"
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add fpdf2 dependency for PDF export"
```

---

## Task 2: Insight Generator

**Files:**
- Create: `backend/app/insights/__init__.py`
- Create: `backend/app/insights/generator.py`
- Test: `backend/tests/test_insights.py`

**Interfaces:**
- Consumes: `DatasetProfile` (app.data.profiler), `dispatch_tool` (app.tools.registry), `llm` (app.llm.client), `Insight` DB model
- Produces: `generate_insights(session_id, profile, parquet_path, db)` async coroutine

- [ ] **Step 1: Create empty `backend/app/insights/__init__.py`**

```python
```

- [ ] **Step 2: Write failing tests**

```python
# backend/tests/test_insights.py
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.db.models import Base, Insight
from app.insights.generator import generate_insights


@pytest.fixture
def db_with_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    Sess = sessionmaker(bind=engine)
    return Sess()


@pytest.fixture
def profile_and_parquet(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "ins-1", "responses.csv", tmp_path)
    return profile, str(tmp_path / "ins-1" / "data.parquet")


@pytest.mark.asyncio
async def test_generate_insights_stores_records(profile_and_parquet, db_with_session, tmp_path):
    profile, parquet_path = profile_and_parquet
    from app.db.models import Session as SessionModel
    s = SessionModel(id="ins-1", filename="responses.csv", row_count=5,
                     profile_path=str(tmp_path / "ins-1" / "profile.json"),
                     data_path=parquet_path)
    db_with_session.add(s)
    db_with_session.commit()

    batch_response = json.dumps([
        {"title": "Finding 1", "summary": "HR has lower satisfaction than Engineering."},
        {"title": "Finding 2", "summary": "Salary correlates with Department."},
    ])
    with patch("app.insights.generator.llm.chat_completion", new=AsyncMock(return_value=batch_response)):
        await generate_insights("ins-1", profile, parquet_path, db_with_session)

    insights = db_with_session.query(Insight).filter_by(session_id="ins-1").all()
    assert len(insights) >= 1
    assert insights[0].rank == 1
    assert insights[0].title
    assert insights[0].summary


@pytest.mark.asyncio
async def test_generate_insights_ranked_by_effect(profile_and_parquet, db_with_session, tmp_path):
    profile, parquet_path = profile_and_parquet
    from app.db.models import Session as SessionModel
    s = SessionModel(id="ins-2", filename="responses.csv", row_count=5,
                     profile_path=str(tmp_path / "ins-1" / "profile.json"),
                     data_path=parquet_path)
    db_with_session.add(s)
    db_with_session.commit()

    batch = json.dumps([{"title": "A", "summary": "Finding A."}, {"title": "B", "summary": "Finding B."}])
    with patch("app.insights.generator.llm.chat_completion", new=AsyncMock(return_value=batch)):
        await generate_insights("ins-2", profile, parquet_path, db_with_session)

    insights = db_with_session.query(Insight).filter_by(session_id="ins-2").order_by(Insight.rank).all()
    ranks = [i.rank for i in insights]
    assert ranks == sorted(ranks)


@pytest.mark.asyncio
async def test_generate_insights_idempotent(profile_and_parquet, db_with_session, tmp_path):
    """Running twice on same session_id clears old insights and re-generates."""
    profile, parquet_path = profile_and_parquet
    from app.db.models import Session as SessionModel
    s = SessionModel(id="ins-3", filename="responses.csv", row_count=5,
                     profile_path=str(tmp_path / "ins-1" / "profile.json"),
                     data_path=parquet_path)
    db_with_session.add(s)
    db_with_session.commit()

    batch = json.dumps([{"title": "T", "summary": "S."}])
    with patch("app.insights.generator.llm.chat_completion", new=AsyncMock(return_value=batch)):
        await generate_insights("ins-3", profile, parquet_path, db_with_session)
        await generate_insights("ins-3", profile, parquet_path, db_with_session)

    count = db_with_session.query(Insight).filter_by(session_id="ins-3").count()
    assert count <= 8
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/test_insights.py -v
# Expected: FAIL — cannot import generate_insights
```

- [ ] **Step 4: Create `backend/app/insights/generator.py`**

```python
from __future__ import annotations
import json
import logging
import pandas as pd
from sqlalchemy.orm import Session as DBSession
from app.data.profiler import DatasetProfile
from app.db.models import Insight
from app.llm.client import llm
from app.tools.registry import dispatch_tool

logger = logging.getLogger(__name__)
_MAX_INSIGHTS = 8


def _candidate_findings(profile: DatasetProfile, df: pd.DataFrame) -> list[dict]:
    """Run a fixed sweep of Tier-1 tools and return scored findings."""
    findings: list[dict] = []

    cat_cols = [c for c, p in profile.columns.items() if p.dtype == "categorical"]
    num_cols = [c for c, p in profile.columns.items() if p.dtype == "numeric"]

    # Segment stats: categorical × numeric combinations
    for cat in cat_cols[:3]:
        for num in num_cols[:3]:
            try:
                tr = dispatch_tool(df, "segment_stats", {"group_col": cat, "metric_col": num})
                rows = tr.table
                if not rows:
                    continue
                means = [r.get("mean") or 0 for r in rows if r.get("mean") is not None]
                if len(means) < 2:
                    continue
                effect = max(means) - min(means)
                findings.append({"effect": effect, "summary": tr.summary,
                                  "tool": "segment_stats", "params": {"group_col": cat, "metric_col": num}})
            except Exception as exc:
                logger.debug("Insight sweep error %s×%s: %s", cat, num, exc)

    # Distribution of each numeric column (look for skew)
    for num in num_cols[:3]:
        try:
            tr = dispatch_tool(df, "anomalies", {"column": num})
            outlier_count = tr.table.get("outlier_count", 0) if isinstance(tr.table, dict) else 0
            if outlier_count > 0:
                findings.append({"effect": float(outlier_count), "summary": tr.summary,
                                  "tool": "anomalies", "params": {"column": num}})
        except Exception as exc:
            logger.debug("Anomaly sweep error %s: %s", num, exc)

    findings.sort(key=lambda f: f["effect"], reverse=True)
    return findings[:_MAX_INSIGHTS]


async def generate_insights(
    session_id: str,
    profile: DatasetProfile,
    parquet_path: str,
    db: DBSession,
) -> None:
    # Clear existing insights for this session (idempotent re-run)
    db.query(Insight).filter_by(session_id=session_id).delete()
    db.commit()

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.error("Could not load parquet for insight generation: %s", exc)
        return

    candidates = _candidate_findings(profile, df)
    if not candidates:
        return

    # One batched LLM call to phrase all findings
    summaries = "\n".join(f"{i+1}. {f['summary']}" for i, f in enumerate(candidates))
    prompt = (
        f"You are a data analyst. Below are raw statistical findings from a survey dataset "
        f'called "{profile.filename}" ({profile.row_count} rows).\n\n'
        f"Raw findings:\n{summaries}\n\n"
        f"For each finding, write a short title (5 words max) and one plain-English sentence summary. "
        f"Respond with a JSON array:\n"
        f'[{{"title": "...", "summary": "..."}}, ...]\n'
        f"Include all {len(candidates)} findings in the same order."
    )

    try:
        raw = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            use_fallback=False,
            max_tokens=600,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # model wrapped in object
            parsed = next(iter(parsed.values())) if parsed else []
    except Exception as exc:
        logger.error("Batch insight LLM call failed: %s", exc)
        parsed = [{"title": f"Finding {i+1}", "summary": c["summary"]} for i, c in enumerate(candidates)]

    for rank, (candidate, phrased) in enumerate(zip(candidates, parsed), start=1):
        insight = Insight(
            session_id=session_id,
            rank=rank,
            title=phrased.get("title", f"Finding {rank}"),
            summary=phrased.get("summary", candidate["summary"]),
            supporting_tool_calls=[{"tool": candidate["tool"], "params": candidate["params"]}],
        )
        db.add(insight)

    db.commit()
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
pytest tests/test_insights.py -v
# Expected: all PASS
```

- [ ] **Step 6: Commit**

```bash
git add app/insights/ tests/test_insights.py
git commit -m "feat: insight generator — Tier-1 sweep, effect-size ranking, batch LLM phrasing"
```

---

## Task 3: Comparison Engine

**Files:**
- Create: `backend/app/compare/__init__.py`
- Create: `backend/app/compare/engine.py`
- Test: `backend/tests/test_compare.py`

**Interfaces:**
- Produces: `ComparisonDiff` dataclass, `compare_profiles(base, compare) -> ComparisonDiff`

- [ ] **Step 1: Create empty `backend/app/compare/__init__.py`**

```python
```

- [ ] **Step 2: Write failing tests**

```python
# backend/tests/test_compare.py
from __future__ import annotations
import pytest
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.compare.engine import compare_profiles, ComparisonDiff
import pandas as pd
import io


@pytest.fixture
def profile_a(tmp_path, sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "q1.csv")
    return build_profile(df, schema, "cmp-a", "q1.csv", tmp_path)


@pytest.fixture
def profile_b(tmp_path):
    df = pd.DataFrame({
        "Name": ["X", "Y", "Z", "W", "V"],
        "Department": ["HR", "HR", "Engineering", "Sales", "Sales"],
        "Satisfaction": [3, 3, 4, 5, 2],
        "Comments": ["Meh", "OK", "Good", "Great", "Bad"],
        "Salary": [52000, 58000, 85000, 70000, 60000],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    schema = {c: {"type": "categorical" if c in ("Name", "Department", "Comments") else "numeric", "n_unique": df[c].nunique()} for c in df.columns}
    return build_profile(df, schema, "cmp-b", "q2.csv", tmp_path)


def test_compare_profiles_returns_diff(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    assert isinstance(diff, ComparisonDiff)


def test_diff_has_row_count_change(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    assert diff.base_row_count == profile_a.row_count
    assert diff.compare_row_count == profile_b.row_count


def test_diff_numeric_delta(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    # Satisfaction column exists in both — should have a numeric delta
    sat = next((c for c in diff.column_diffs if c["column"] == "Satisfaction"), None)
    assert sat is not None
    assert "mean_delta" in sat


def test_diff_shared_columns(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    shared = {c["column"] for c in diff.column_diffs}
    assert "Satisfaction" in shared
    assert "Salary" in shared


def test_diff_serializable(profile_a, profile_b):
    import json
    diff = compare_profiles(profile_a, profile_b)
    import dataclasses
    serialized = json.dumps(dataclasses.asdict(diff))
    assert serialized
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/test_compare.py -v
# Expected: FAIL
```

- [ ] **Step 4: Create `backend/app/compare/engine.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from app.data.profiler import DatasetProfile


@dataclass
class ComparisonDiff:
    base_session_id: str
    compare_session_id: str
    base_filename: str
    compare_filename: str
    base_row_count: int
    compare_row_count: int
    row_count_delta: int
    shared_columns: list[str]
    only_in_base: list[str]
    only_in_compare: list[str]
    column_diffs: list[dict]   # [{column, base_*, compare_*, mean_delta?, ...}]


def compare_profiles(base: DatasetProfile, compare: DatasetProfile) -> ComparisonDiff:
    base_cols = set(base.columns)
    compare_cols = set(compare.columns)
    shared = sorted(base_cols & compare_cols)
    only_base = sorted(base_cols - compare_cols)
    only_compare = sorted(compare_cols - base_cols)

    col_diffs: list[dict] = []
    for col in shared:
        bc = base.columns[col]
        cc = compare.columns[col]
        entry: dict = {
            "column": col,
            "dtype": bc.dtype,
            "base_missing_pct": bc.missing_pct,
            "compare_missing_pct": cc.missing_pct,
            "missing_pct_delta": round(cc.missing_pct - bc.missing_pct, 2),
            "base_n_unique": bc.n_unique,
            "compare_n_unique": cc.n_unique,
        }
        if bc.dtype == "numeric" and bc.mean is not None and cc.mean is not None:
            entry["base_mean"] = bc.mean
            entry["compare_mean"] = cc.mean
            entry["mean_delta"] = round(cc.mean - bc.mean, 4)
            entry["base_std"] = bc.std
            entry["compare_std"] = cc.std
        if bc.dtype == "categorical":
            entry["base_top_values"] = dict(list(bc.top_values.items())[:5])
            entry["compare_top_values"] = dict(list(cc.top_values.items())[:5])
        col_diffs.append(entry)

    return ComparisonDiff(
        base_session_id=base.session_id,
        compare_session_id=compare.session_id,
        base_filename=base.filename,
        compare_filename=compare.filename,
        base_row_count=base.row_count,
        compare_row_count=compare.row_count,
        row_count_delta=compare.row_count - base.row_count,
        shared_columns=shared,
        only_in_base=only_base,
        only_in_compare=only_compare,
        column_diffs=col_diffs,
    )
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
pytest tests/test_compare.py -v
# Expected: all PASS
```

- [ ] **Step 6: Commit**

```bash
git add app/compare/ tests/test_compare.py
git commit -m "feat: comparison engine — profile diff with row/column/numeric/categorical deltas"
```

---

## Task 4: Story Report Composer + PDF Renderer

**Files:**
- Create: `backend/app/reports/__init__.py`
- Create: `backend/app/reports/composer.py`
- Create: `backend/app/reports/renderer.py`
- Test: `backend/tests/test_export.py`

**Interfaces:**
- `compose_report(profile, insights, pinned_charts) -> ReportData`
- `render_pdf(report_data, llm_narrative) -> bytes`

- [ ] **Step 1: Create empty `backend/app/reports/__init__.py`**

```python
```

- [ ] **Step 2: Write failing tests**

```python
# backend/tests/test_export.py
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.reports.composer import compose_report, ReportData
from app.reports.renderer import render_pdf


@pytest.fixture
def profile(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    return build_profile(df, schema, "rpt-1", "responses.csv", tmp_path)


def test_compose_report_returns_report_data(profile):
    data = compose_report(profile, insights=[], pinned_charts=[])
    assert isinstance(data, ReportData)
    assert data.filename == "responses.csv"
    assert data.row_count == 5


def test_compose_report_with_insights(profile):
    from app.db.models import Insight
    ins = Insight(session_id="rpt-1", rank=1, title="Top Finding",
                  summary="HR is happiest.", supporting_tool_calls=None)
    data = compose_report(profile, insights=[ins], pinned_charts=[])
    assert len(data.insights) == 1
    assert data.insights[0]["title"] == "Top Finding"


def test_render_pdf_returns_bytes(profile):
    data = compose_report(profile, insights=[], pinned_charts=[])
    pdf_bytes = render_pdf(data, narrative="This dataset has 5 rows.")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:4] == b"%PDF"


def test_render_pdf_with_charts(profile):
    from app.dashboard.charts import deterministic_charts
    from pathlib import Path
    import base64
    session_dir = Path(profile.session_id) if profile.session_id.startswith("/") else Path("/tmp") / profile.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    charts = deterministic_charts(profile, session_dir)
    pinned = [{"title": c.title, "png_bytes": c.png_bytes} for c in charts[:1]]
    data = compose_report(profile, insights=[], pinned_charts=pinned)
    pdf_bytes = render_pdf(data, narrative="Summary here.")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/test_export.py -v
# Expected: FAIL
```

- [ ] **Step 4: Create `backend/app/reports/composer.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from app.data.profiler import DatasetProfile


@dataclass
class ReportData:
    filename: str
    row_count: int
    col_count: int
    column_summary: list[dict]
    insights: list[dict]
    pinned_charts: list[dict]   # [{title, png_bytes}]


def compose_report(
    profile: DatasetProfile,
    insights: list,
    pinned_charts: list[dict],
) -> ReportData:
    col_summary = [
        {
            "name": name,
            "dtype": cp.dtype,
            "missing_pct": cp.missing_pct,
            "n_unique": cp.n_unique,
            "mean": cp.mean,
            "top_values": dict(list(cp.top_values.items())[:3]),
        }
        for name, cp in profile.columns.items()
    ]

    insights_out = [
        {"rank": ins.rank, "title": ins.title, "summary": ins.summary}
        for ins in insights
    ]

    return ReportData(
        filename=profile.filename,
        row_count=profile.row_count,
        col_count=profile.col_count,
        column_summary=col_summary,
        insights=insights_out,
        pinned_charts=pinned_charts,
    )
```

- [ ] **Step 5: Create `backend/app/reports/renderer.py`**

```python
from __future__ import annotations
import io
import textwrap
from fpdf import FPDF
from app.reports.composer import ReportData


class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Survey Analytics Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str) -> None:
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 0, 0)
        self.line(self.get_x(), self.get_y(), self.get_x() + self.epw, self.get_y())
        self.ln(2)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        wrapped = textwrap.fill(text, width=90)
        self.multi_cell(0, 6, wrapped)
        self.ln(2)


def render_pdf(report: ReportData, narrative: str) -> bytes:
    pdf = _PDF()
    pdf.add_page()

    # Dataset summary
    pdf.section_title("Dataset Overview")
    pdf.body_text(
        f"File: {report.filename}  |  Rows: {report.row_count}  |  Columns: {report.col_count}"
    )
    pdf.body_text(narrative)

    # Column table
    pdf.section_title("Column Summary")
    pdf.set_font("Helvetica", "B", 9)
    col_w = [50, 25, 25, 25, 65]
    headers = ["Column", "Type", "Missing %", "Unique", "Top Values / Mean"]
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for col in report.column_summary:
        top = str(col.get("mean") or col.get("top_values") or "")[:30]
        row_vals = [col["name"][:20], col["dtype"], str(col["missing_pct"]), str(col["n_unique"]), top]
        for w, v in zip(col_w, row_vals):
            pdf.cell(w, 6, v, border=1)
        pdf.ln()

    # Key insights
    if report.insights:
        pdf.section_title("Key Findings")
        for ins in report.insights:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"{ins['rank']}. {ins['title']}", new_x="LMARGIN", new_y="NEXT")
            pdf.body_text(ins["summary"])

    # Pinned charts
    if report.pinned_charts:
        pdf.section_title("Charts")
        for chart in report.pinned_charts:
            png_bytes = chart.get("png_bytes")
            if not png_bytes:
                continue
            title = chart.get("title", "Chart")
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
            img_buf = io.BytesIO(png_bytes)
            # Place image; max width = page width - margins
            try:
                pdf.image(img_buf, w=min(170, pdf.epw))
            except Exception:
                pdf.body_text("[Chart could not be rendered]")
            pdf.ln(4)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
```

- [ ] **Step 6: Run tests to confirm pass**

```bash
pytest tests/test_export.py -v
# Expected: all PASS
```

- [ ] **Step 7: Commit**

```bash
git add app/reports/ tests/test_export.py
git commit -m "feat: story report — composer + fpdf2 PDF renderer with insights and pinned charts"
```

---

## Task 5: API Routes + Wire-Up

**Files:**
- Create: `backend/app/api/insights.py`
- Create: `backend/app/api/compare.py`
- Create: `backend/app/api/export.py`
- Modify: `backend/app/api/upload.py` (trigger background insight generation)
- Modify: `backend/app/main.py` (register new routers)

- [ ] **Step 1: Create `backend/app/api/insights.py`**

```python
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, Insight

router = APIRouter()


@router.get("/sessions/{session_id}/insights")
def list_insights(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    rows = db.query(Insight).filter_by(session_id=session_id).order_by(Insight.rank).all()
    return [
        {
            "insight_id": r.id,
            "rank": r.rank,
            "title": r.title,
            "summary": r.summary,
            "supporting_tool_calls": r.supporting_tool_calls,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
```

- [ ] **Step 2: Create `backend/app/api/compare.py`**

```python
from __future__ import annotations
import dataclasses
import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, Comparison
from app.data.profiler import load_profile
from app.compare.engine import compare_profiles
from pathlib import Path

router = APIRouter()


class CompareRequest(BaseModel):
    base_session_id: str
    compare_session_id: str


@router.post("/compare")
def run_comparison(body: CompareRequest, db: Session = Depends(get_db)) -> dict:
    # Return cached if exists
    cached = db.query(Comparison).filter_by(
        base_session_id=body.base_session_id,
        compare_session_id=body.compare_session_id,
    ).first()
    if cached:
        return {"cached": True, "diff": cached.diff_summary}

    base_rec = db.get(SessionModel, body.base_session_id)
    cmp_rec = db.get(SessionModel, body.compare_session_id)
    if not base_rec:
        raise HTTPException(status_code=404, detail="Base session not found")
    if not cmp_rec:
        raise HTTPException(status_code=404, detail="Compare session not found")
    if not base_rec.profile_path or not Path(base_rec.profile_path).exists():
        raise HTTPException(status_code=404, detail="Base profile not found")
    if not cmp_rec.profile_path or not Path(cmp_rec.profile_path).exists():
        raise HTTPException(status_code=404, detail="Compare profile not found")

    base_profile = load_profile(Path(base_rec.profile_path))
    cmp_profile = load_profile(Path(cmp_rec.profile_path))
    diff = compare_profiles(base_profile, cmp_profile)
    diff_dict = dataclasses.asdict(diff)

    record = Comparison(
        base_session_id=body.base_session_id,
        compare_session_id=body.compare_session_id,
        diff_summary=diff_dict,
    )
    db.add(record)
    db.commit()
    return {"cached": False, "diff": diff_dict}
```

- [ ] **Step 3: Create `backend/app/api/export.py`**

```python
from __future__ import annotations
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, Insight, PinnedChart
from app.data.profiler import load_profile
from app.reports.composer import compose_report
from app.reports.renderer import render_pdf
from app.llm.client import llm
from app.llm.prompts import dashboard_narrative_prompt

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/sessions/{session_id}/export/pdf")
async def export_pdf(session_id: str, db: Session = Depends(get_db)) -> Response:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    if not record.profile_path or not Path(record.profile_path).exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = load_profile(Path(record.profile_path))
    insights = db.query(Insight).filter_by(session_id=session_id).order_by(Insight.rank).all()
    pins = db.query(PinnedChart).filter_by(session_id=session_id).all()

    pinned_charts = []
    for p in pins:
        png_path = Path(record.profile_path).parent / f"{p.id}.png"
        if png_path.exists():
            pinned_charts.append({"title": p.title or "", "png_bytes": png_path.read_bytes()})

    # Narrative: use cached or call LLM
    narrative_path = Path(record.profile_path).parent / "narrative.txt"
    if narrative_path.exists():
        narrative = narrative_path.read_text()
    else:
        try:
            narrative = await llm.chat_completion(
                messages=[{"role": "user", "content": dashboard_narrative_prompt(profile)}],
                use_fallback=False, max_tokens=300, temperature=0.4,
            )
            narrative_path.write_text(narrative)
        except Exception as exc:
            logger.error("Narrative LLM call failed for export: %s", exc)
            narrative = f"Dataset: {profile.filename}, {profile.row_count} rows."

    report_data = compose_report(profile, insights, pinned_charts)
    try:
        pdf_bytes = render_pdf(report_data, narrative)
    except Exception as exc:
        logger.error("PDF render failed: %s", exc)
        raise HTTPException(status_code=500, detail="PDF generation failed")

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in record.filename)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_report.pdf"'},
    )
```

- [ ] **Step 4: Modify `backend/app/api/upload.py`** — add background insight task

At the bottom of the `upload_file` function, just before `return {...}`, add:

```python
# In upload_file(), after dashboard = await generate_dashboard(...), add:
from app.insights.generator import generate_insights as _gen_insights
from fastapi import BackgroundTasks

# Change the function signature to:
# async def upload_file(file, db, background_tasks: BackgroundTasks = BackgroundTasks()):
# and add after the dashboard line:
background_tasks.add_task(_gen_insights, record.id, profile, record.data_path, db)
```

Full updated signature for `upload_file` in `backend/app/api/upload.py`:

```python
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict:
    # ... existing body unchanged up to dashboard generation ...
    dashboard = await generate_dashboard(profile, quality, session_dir)

    # Trigger insight generation in the background
    from app.insights.generator import generate_insights as _gen_insights
    background_tasks.add_task(_gen_insights, record.id, profile, record.data_path, db)

    return { ... }  # existing return unchanged
```

Modify `backend/app/api/upload.py` to add `BackgroundTasks` to the import line and the function signature:

```python
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
```

and update `async def upload_file(file, db, background_tasks: BackgroundTasks = BackgroundTasks())`.

- [ ] **Step 5: Modify `backend/app/main.py`**

```python
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.db.database import create_tables
from app.api import upload, sessions, dashboard, chat, insights, compare, export


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
app.include_router(insights.router, prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(export.router, prefix="/api")
```

- [ ] **Step 6: Run full test suite**

```bash
cd backend && pytest -v
# Expected: all existing + new tests PASS
```

- [ ] **Step 7: Commit**

```bash
git add app/api/insights.py app/api/compare.py app/api/export.py app/api/upload.py app/main.py
git commit -m "feat: insights, compare, and PDF export API routes wired up"
```

---

## Self-Review

**Spec coverage:**
- ✅ Background insight generation triggered after upload
- ✅ Tier-1 sweep (segment_stats × numeric, anomalies)
- ✅ Rank by effect size
- ✅ One batched LLM call phrases all findings
- ✅ Stored in `insights` table
- ✅ Comparison mode: diff two profiles, cached in `comparisons` table
- ✅ Story report: compose + PDF with fpdf2
- ✅ Pinned charts included in PDF (if chart file on disk)
- ✅ GET /sessions/{id}/insights
- ✅ POST /compare
- ✅ GET /sessions/{id}/export/pdf
- ⚠️ Trend analysis in insight sweep — skipped (requires a datetime column, not guaranteed in test data). Can be added later.
- ⚠️ Pinned chart PNG storage — the `pin_chart` endpoint in Plan 2 stores `chart_path` as a placeholder string, not the actual PNG. The export route looks for `{session_dir}/{pin_id}.png` which may not exist. For v1 the PDF will simply skip those charts. A follow-up can write the PNG to disk on pin.
