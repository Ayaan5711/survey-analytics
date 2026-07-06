from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def _read_any(filename: str, raw: bytes) -> list[pd.DataFrame]:
    """Read a CSV/Excel upload into one or more frames (one per sheet)."""
    if filename.lower().endswith(".csv"):
        return [pd.read_csv(io.BytesIO(raw))]
    book = pd.read_excel(io.BytesIO(raw), sheet_name=None)  # all sheets
    return [df for df in book.values() if not df.empty]


def _sig(df: pd.DataFrame) -> tuple:
    """Schema signature = normalized column names (to group same-schema parts)."""
    return tuple(str(c).strip().lower() for c in df.columns)


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


@dataclass
class SurveySession:
    session_id: str
    filename: str                 # combined label, e.g. "a.xlsx + b.xlsx"
    files: list[str]              # individual uploaded file names
    df: pd.DataFrame              # all files combined (in memory, like the original design)
    profile: str                  # compact text profile for the LLM
    chat_history: list[dict[str, str]] = field(default_factory=list)

    # Kept for frontend compatibility (his UI expects these names).
    @property
    def sheet_names(self) -> list[str]:
        return self.files

    @property
    def active_sheet(self) -> str:
        return self.filename

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.df.columns]

    @property
    def row_count(self) -> int:
        return len(self.df)

    def dataframe(self) -> pd.DataFrame:
        return self.df


class SurveySessionStore:
    """In-memory session store (single gunicorn worker) — same model as the
    original ConvInsight design, extended to combine multiple same-schema files."""

    def __init__(self) -> None:
        self._sessions: dict[str, SurveySession] = {}

    def create_from_uploads(self, uploads: list[tuple[str, bytes]]) -> SurveySession:
        """uploads = [(filename, bytes), ...]. Same-schema parts are combined into one dataset."""
        if not uploads:
            raise ValueError("No files provided.")

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

        combined = pd.concat([d for _, d in chosen], ignore_index=True)
        file_names = list(dict.fromkeys(fname for fname, _ in chosen))  # de-duplicated, order preserved

        session_id = str(uuid.uuid4())
        profile = _build_profile(combined, file_names, len(combined))
        session = SurveySession(
            session_id=session_id, filename=" + ".join(file_names), files=file_names,
            df=combined, profile=profile,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SurveySession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    @staticmethod
    def dataframe_preview(session: SurveySession, max_rows: int = 8) -> list[dict[str, Any]]:
        return session.df.head(max_rows).fillna("").to_dict(orient="records")
