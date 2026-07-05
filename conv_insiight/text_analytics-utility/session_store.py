from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


# Local disk only (no infra). Placeholder path; override with CONVINSIGHT_DATA if needed.
DATA_DIR = Path(os.getenv("CONVINSIGHT_DATA", str(Path(tempfile.gettempdir()) / "convinsight_data")))


def _read_any(filename: str, raw: bytes) -> list[pd.DataFrame]:
    """Read a CSV/Excel upload into one or more frames (one per sheet)."""
    if filename.lower().endswith(".csv"):
        return [pd.read_csv(io.BytesIO(raw))]
    book = pd.read_excel(io.BytesIO(raw), sheet_name=None)  # all sheets
    return [df for df in book.values() if not df.empty]


def _sig(df: pd.DataFrame) -> tuple:
    """Schema signature = normalized column names (to group same-schema parts)."""
    return tuple(str(c).strip().lower() for c in df.columns)


@dataclass
class SurveySession:
    session_id: str
    filename: str                 # combined label, e.g. "a.xlsx + b.xlsx"
    files: list[str]              # individual uploaded file names
    columns: list[str]            # shared schema
    row_count: int
    profile: str                  # compact text profile for the LLM
    dir: Path
    con: duckdb.DuckDBPyConnection
    chat_history: list[dict[str, str]] = field(default_factory=list)

    # Kept for frontend compatibility (his UI expects these names).
    @property
    def sheet_names(self) -> list[str]:
        return self.files

    @property
    def active_sheet(self) -> str:
        return self.filename

    def dataframe(self) -> pd.DataFrame:
        """Materialize the combined dataset (all files unioned)."""
        return self.con.execute("SELECT * FROM data").df()


def _build_profile(df: pd.DataFrame, files: list[str], row_count: int) -> str:
    lines = [f"Dataset: {' + '.join(files)} — {row_count} rows (all files combined), {df.shape[1]} columns.",
             "Columns:"]
    for col in df.columns:
        s = df[col]
        miss = round(s.isna().mean() * 100, 1)
        if pd.api.types.is_numeric_dtype(s):
            mean = round(float(s.mean()), 2) if s.notna().any() else None
            lines.append(f"- {col} (numeric): min={s.min()}, max={s.max()}, mean={mean}, missing={miss}%")
        else:
            top = s.dropna().astype(str).value_counts().head(6)
            vals = "; ".join(f'"{k}"' for k in top.index)
            lines.append(f'- {col} (categorical, {int(s.nunique())} values; e.g. {vals}), missing={miss}%')
    return "\n".join(lines)


class SurveySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SurveySession] = {}
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def create_from_uploads(self, uploads: list[tuple[str, bytes]]) -> SurveySession:
        """uploads = [(filename, bytes), ...]. Same-schema parts are unioned into one dataset."""
        if not uploads:
            raise ValueError("No files provided.")

        # Read every sheet of every file, then keep the majority schema group.
        parts: list[tuple[str, pd.DataFrame]] = []
        for fname, raw in uploads:
            for df in _read_any(fname, raw):
                parts.append((fname, df))
        if not parts:
            raise ValueError("Uploaded file(s) had no readable data.")

        groups: dict[tuple, list[tuple[str, pd.DataFrame]]] = {}
        for fname, df in parts:
            groups.setdefault(_sig(df), []).append((fname, df))
        best_sig = max(groups, key=lambda s: sum(len(d) for _, d in groups[s]))
        chosen = groups[best_sig]

        session_id = str(uuid.uuid4())
        sess_dir = DATA_DIR / session_id
        sess_dir.mkdir(parents=True, exist_ok=True)

        parquet_paths: list[str] = []
        file_names: list[str] = []
        for idx, (fname, df) in enumerate(chosen):
            p = sess_dir / f"part_{idx}.parquet"
            df.to_parquet(p, index=False)
            parquet_paths.append(str(p))
            if fname not in file_names:
                file_names.append(fname)

        con = duckdb.connect()  # in-process, no server
        path_list = ", ".join(f"'{p}'" for p in parquet_paths)
        con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet([{path_list}])")
        row_count = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]

        sample = con.execute("SELECT * FROM data LIMIT 20000").df()  # bounded scan for profile
        columns = [str(c) for c in sample.columns]
        profile = _build_profile(sample, file_names, row_count)

        meta = {"files": file_names, "parquet": [Path(p).name for p in parquet_paths],
                "columns": columns, "row_count": row_count, "profile": profile}
        (sess_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        session = SurveySession(
            session_id=session_id, filename=" + ".join(file_names), files=file_names,
            columns=columns, row_count=row_count, profile=profile, dir=sess_dir, con=con,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SurveySession:
        """Return a live session, reloading from disk if the process restarted."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        sess_dir = DATA_DIR / session_id
        meta_path = sess_dir / "meta.json"
        if not meta_path.exists():
            raise KeyError(f"Unknown session_id: {session_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        con = duckdb.connect()
        paths = ", ".join(f"'{sess_dir / p}'" for p in meta["parquet"])
        con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet([{paths}])")
        session = SurveySession(
            session_id=session_id, filename=" + ".join(meta["files"]), files=meta["files"],
            columns=meta["columns"], row_count=meta["row_count"], profile=meta["profile"],
            dir=sess_dir, con=con,
        )
        self._sessions[session_id] = session
        return session

    @staticmethod
    def dataframe_preview(session: SurveySession, max_rows: int = 8) -> list[dict[str, Any]]:
        df = session.con.execute(f"SELECT * FROM data LIMIT {max_rows}").df()
        return df.fillna("").to_dict(orient="records")
