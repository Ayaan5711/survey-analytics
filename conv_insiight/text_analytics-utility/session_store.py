from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_SESSION_TTL_SECONDS = 4 * 60 * 60  # evict sessions idle this long (bounds memory over long uptime)
_MAX_CHAT_HISTORY = 40              # keep a bit more than the agent actually reads, then trim


def _read_any(filename: str, raw: bytes) -> list[tuple[str, pd.DataFrame]]:
    """Read a CSV/Excel upload into one or more (label, frame) parts.

    label is just the filename for CSV / single-sheet Excel, or
    "filename — SheetName" for each sheet of a multi-sheet workbook — this
    keeps every part uniquely identifiable so a mismatched sheet can be
    reported by name instead of being confused with a same-named sibling
    sheet from the same file (a real filename collision otherwise makes a
    dropped sheet impossible to distinguish from a kept one).
    """
    if filename.lower().endswith(".csv"):
        return [(filename, pd.read_csv(io.BytesIO(raw)))]
    book = pd.read_excel(io.BytesIO(raw), sheet_name=None)  # all sheets
    non_empty = {name: df for name, df in book.items() if not df.empty}
    if len(non_empty) <= 1:
        return [(filename, df) for df in non_empty.values()]
    return [(f"{filename} — {sheet}", df) for sheet, df in non_empty.items()]


_MERGE_SIMILARITY_THRESHOLD = 0.85  # Jaccard similarity of normalized column names
                                    # to still count as "almost identical" and merge
                                    # (e.g. tolerates a couple of extra/missing columns
                                    # on a 97-column survey; well below that is a
                                    # genuinely different file, not a near-match).


def _norm_col(c) -> str:
    return str(c).strip().lower()


def _col_set(df: pd.DataFrame) -> frozenset:
    """Schema signature = normalized column names, order-independent (so a
    reordered-but-identical schema still counts as an exact match)."""
    return frozenset(_norm_col(c) for c in df.columns)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _build_profile(df: pd.DataFrame, files: list[str], row_count: int) -> str:
    """Full profile: columns + exact sample top-values + numeric stats, sent in
    the system prompt on every chat turn.

    NOTE: this deliberately matches the original survey app's own
    _profile_summary() (backend/app/llm/prompts.py) — that app's proven
    design keeps exact categorical values in the prompt every turn (its own
    comment: "so the model passes them verbatim to tools") rather than
    compacting them away. An earlier version of this file tried a much
    smaller name+type-only profile to cut tokens, with sample values only
    available via the on-demand dataset_schema tool — reverted after
    review, since it deviated from the master app's tested behavior and
    risked the LLM guessing at value spellings instead of using the real
    ones. Correctness over marginal token savings here.
    """
    lines = [f"Dataset: {' + '.join(files)} — {row_count} rows (all files combined), {df.shape[1]} columns.",
             "Columns:"]
    for col in df.columns:
        s = df[col]
        miss = round(s.isna().mean() * 100, 1)
        if pd.api.types.is_numeric_dtype(s):
            mean = round(float(s.mean()), 2) if s.notna().any() else None
            lines.append(f"- {col} (numeric): min={s.min()}, max={s.max()}, mean={mean}, missing={miss}%")
        else:
            top = s.dropna().astype(str).value_counts().head(8)
            vals = "; ".join(f'"{k}"' for k in top.index)
            lines.append(f'- {col} (categorical, {int(s.nunique())} values; exact values: {vals}), missing={miss}%')
    return "\n".join(lines)


@dataclass
class SurveySession:
    session_id: str
    filename: str                 # combined label, e.g. "a.xlsx + b.xlsx"
    files: list[str]              # individual uploaded file names actually combined
    sheets: list[str]             # individual parts combined — "file.xlsx" or "file.xlsx — SheetName"
    df: pd.DataFrame              # all files combined (in memory, like the original design)
    profile: str                  # full profile (columns, types, exact top-values) — sent every chat turn
    skipped_files: list[str] = field(default_factory=list)  # schema mismatch — not combined (by part label)
    skipped_detail: list[dict] = field(default_factory=list)  # [{label, columns, match_pct}] -- why each was skipped
    merge_detail: list[dict] = field(default_factory=list)  # [{label, exact_match, missing_columns, extra_columns}]
    chat_history: list[dict[str, str]] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    answer_cache: dict[str, dict] = field(default_factory=dict)   # question (normalized) -> {answer, charts}
    dashboard_cache: dict | None = None                            # computed once at upload, reused on demand

    def touch(self) -> None:
        self.last_active = time.time()

    def add_message(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > _MAX_CHAT_HISTORY:
            # Bound memory for very long-running chats; the agent only ever
            # reads the last ~16 messages anyway.
            self.chat_history = self.chat_history[-_MAX_CHAT_HISTORY:]

    # Kept for frontend compatibility (his UI expects these names) — now backed
    # by real per-sheet labels instead of just the uploaded filenames.
    @property
    def sheet_names(self) -> list[str]:
        return self.sheets

    @property
    def active_sheet(self) -> str:
        return self.sheets[0] if self.sheets else self.filename

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
        """uploads = [(filename, bytes), ...].

        Every part (each CSV, or each sheet of an Excel file) gets an explicit
        decision, all surfaced to the caller (no silent drops):
          - exact schema match to the majority  -> merged
          - "almost identical" (>= _MERGE_SIMILARITY_THRESHOLD column overlap)
            -> merged too, aligned by column name; any columns one side lacks
            just come out NaN for that side's rows (pandas concat's normal
            union-of-columns behavior)
          - below the threshold -> skipped, with the actual match% and column
            count so the reason is concrete, not just "different structure"
        """
        if not uploads:
            raise ValueError("No files provided.")

        parts: list[tuple[str, str, pd.DataFrame]] = []  # (source_filename, part_label, df)
        for fname, raw in uploads:
            for label, df in _read_any(fname, raw):
                parts.append((fname, label, df))
        if not parts:
            raise ValueError("Uploaded file(s) had no readable data.")

        col_sets = {label: _col_set(df) for _, label, df in parts}

        # Exact-schema groups first (order-independent) -- the base is whichever
        # exact group has the most total rows.
        exact_groups: dict[frozenset, list[tuple[str, str, pd.DataFrame]]] = {}
        for fname, label, df in parts:
            exact_groups.setdefault(col_sets[label], []).append((fname, label, df))
        base_set = max(exact_groups, key=lambda s: sum(len(d) for _, _, d in exact_groups[s]))
        base_group = exact_groups[base_set]
        base_cols = list(base_group[0][2].columns)  # canonical original casing/order
        norm_to_canonical = {_norm_col(c): c for c in base_cols}

        merge_detail: list[dict] = []
        skipped: list[str] = []
        skipped_detail: list[dict] = []
        to_merge: list[tuple[str, str, pd.DataFrame]] = []

        for fname, label, df in parts:
            if col_sets[label] == base_set:
                to_merge.append((fname, label, df))
                merge_detail.append({"label": label, "exact_match": True,
                                      "missing_columns": [], "extra_columns": []})
                continue
            similarity = _jaccard(col_sets[label], base_set)
            if similarity >= _MERGE_SIMILARITY_THRESHOLD:
                missing = [c for c in base_cols if _norm_col(c) not in col_sets[label]]
                extra = [c for c in df.columns if _norm_col(c) not in norm_to_canonical]
                # Align this part's columns to the base's canonical names so
                # concat unions correctly instead of treating a case/whitespace
                # difference as a whole separate column.
                renamed = df.rename(columns={c: norm_to_canonical[_norm_col(c)]
                                              for c in df.columns if _norm_col(c) in norm_to_canonical})
                to_merge.append((fname, label, renamed))
                merge_detail.append({"label": label, "exact_match": False,
                                      "missing_columns": missing, "extra_columns": extra,
                                      "match_pct": round(similarity * 100, 1)})
            else:
                skipped.append(label)
                skipped_detail.append({"label": label, "columns": len(col_sets[label]),
                                        "match_pct": round(similarity * 100, 1)})

        combined = pd.concat([d for _, _, d in to_merge], ignore_index=True, sort=False)
        file_names = list(dict.fromkeys(fname for fname, _, _ in to_merge))
        part_labels = list(dict.fromkeys(label for _, label, _ in to_merge))

        self._evict_stale()

        session_id = str(uuid.uuid4())
        profile = _build_profile(combined, file_names, len(combined))
        session = SurveySession(
            session_id=session_id, filename=" + ".join(file_names), files=file_names,
            sheets=part_labels, df=combined, profile=profile, skipped_files=skipped,
            skipped_detail=skipped_detail, merge_detail=merge_detail,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SurveySession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        session.touch()
        return session

    def _evict_stale(self) -> None:
        """Drop sessions idle longer than the TTL — bounds memory over long
        server uptime now that a session can hold a 1M-row dataframe."""
        cutoff = time.time() - _SESSION_TTL_SECONDS
        stale = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in stale:
            del self._sessions[sid]

    @staticmethod
    def dataframe_preview(session: SurveySession, max_rows: int = 8) -> list[dict[str, Any]]:
        return session.df.head(max_rows).fillna("").to_dict(orient="records")
