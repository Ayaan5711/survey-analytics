from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SurveySession:
    session_id: str
    filename: str
    sheet_names: list[str]
    active_sheet: str
    dataframes: dict[str, pd.DataFrame]
    chat_history: list[dict[str, str]] = field(default_factory=list)


class SurveySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SurveySession] = {}

    def create_from_excel_bytes(self, filename: str, file_bytes: bytes) -> SurveySession:
        excel_book = pd.read_excel(file_bytes, sheet_name=None)
        if not excel_book:
            raise ValueError("Excel file has no readable sheets.")

        clean_dataframes: dict[str, pd.DataFrame] = {}
        for sheet, dataframe in excel_book.items():
            clean_dataframes[sheet] = dataframe.copy()

        first_sheet = list(clean_dataframes.keys())[0]
        session_id = str(uuid.uuid4())
        session = SurveySession(
            session_id=session_id,
            filename=filename,
            sheet_names=list(clean_dataframes.keys()),
            active_sheet=first_sheet,
            dataframes=clean_dataframes,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SurveySession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    @staticmethod
    def dataframe_preview(df: pd.DataFrame, max_rows: int = 8) -> list[dict[str, Any]]:
        return df.head(max_rows).fillna("").to_dict(orient="records")
