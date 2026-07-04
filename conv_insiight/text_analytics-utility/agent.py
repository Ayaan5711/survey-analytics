from __future__ import annotations

import json
import logging
import os
from typing import Any

try:
    from langchain_core.tools import StructuredTool
except ImportError:
    from langchain.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from code_executor import run_python_analysis_code
from session_store import SurveySession


logger = logging.getLogger("survey_agent")


def _truncate(value: Any, max_len: int = 1200) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


class PythonCodeInput(BaseModel):
    code: str = Field(
        description=(
            "Python code to run against survey data. Use `df` for active sheet, "
            "`dataframes` for all sheets, and assign answer artifacts to `result`."
        )
    )


class SelectSheetInput(BaseModel):
    sheet_name: str = Field(description="Sheet name to make active for analysis.")


class SurveyAnalysisAgent:
    def __init__(self) -> None:
        # deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        # endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # api_key = os.getenv("AZURE_OPENAI_API_KEY")
        # api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

        deployment = "gpt-5.4"
        endpoint = "https://tcsion-nonprod-resource.cognitiveservices.azure.com"
        api_key = ""
        api_version = "2025-01-01-preview"

        if not deployment or not endpoint or not api_key:
            raise RuntimeError(
                "Missing Azure OpenAI settings. Required: AZURE_OPENAI_DEPLOYMENT, "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"
            )

        self.llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment,
            temperature=0.7,
        )
        logger.info(
            "agent_llm_mode=azure_openai endpoint=%s deployment=%s api_version=%s",
            endpoint,
            deployment,
            api_version,
        )
        self.max_history_messages = int(os.getenv("CHAT_HISTORY_MESSAGES", "16"))

    def _log_agent_messages(self, messages: list[Any]) -> None:
        for index, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = _truncate(getattr(msg, "content", ""), max_len=1000)
            logger.info("trace_step=%s message_type=%s content=%s", index, msg_type, content)

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    logger.info(
                        "trace_step=%s tool_call name=%s args=%s",
                        index,
                        call.get("name"),
                        _truncate(call.get("args"), max_len=1000),
                    )

    def _build_tools(self, session: SurveySession) -> list[StructuredTool]:
        def dataset_schema() -> str:
            logger.info(
                "tool_call=dataset_schema active_sheet=%s sheets=%s",
                session.active_sheet,
                ",".join(session.sheet_names),
            )
            payload: dict[str, Any] = {
                "filename": session.filename,
                "active_sheet": session.active_sheet,
                "sheets": {},
            }
            for sheet_name, df in session.dataframes.items():
                payload["sheets"][sheet_name] = {
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "columns": [str(c) for c in df.columns],
                    "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                    "missing_per_column": {
                        str(k): int(v) for k, v in df.isna().sum().to_dict().items()
                    },
                }
            result = json.dumps(payload)
            logger.info("tool_result=dataset_schema payload=%s", _truncate(result))
            return result

        def select_sheet(sheet_name: str) -> str:
            logger.info("tool_call=select_sheet requested=%s", sheet_name)
            if sheet_name not in session.dataframes:
                result = (
                    "Invalid sheet name. Available sheets: "
                    + ", ".join(session.sheet_names)
                )
                logger.warning("tool_result=select_sheet status=invalid detail=%s", result)
                return result
            session.active_sheet = sheet_name
            result = f"Active sheet changed to '{sheet_name}'."
            logger.info("tool_result=select_sheet status=ok active_sheet=%s", session.active_sheet)
            return result

        def run_python(code: str) -> str:
            logger.info("tool_call=run_python active_sheet=%s", session.active_sheet)
            logger.info("tool_input=run_python code=\n%s", code)
            response = run_python_analysis_code(
                code=code,
                dataframes=session.dataframes,
                active_sheet=session.active_sheet,
            )
            serialized = json.dumps(response, default=str)
            logger.info("tool_result=run_python response=%s", _truncate(serialized, max_len=2000))
            return serialized

        return [
            StructuredTool.from_function(
                name="dataset_schema",
                func=dataset_schema,
                description=(
                    "Get workbook schema details (sheet names, columns, dtypes, shape, missing values). "
                    "Call this before deep analysis."
                ),
            ),
            StructuredTool.from_function(
                name="select_sheet",
                func=select_sheet,
                description="Switch active sheet before analysis.",
                args_schema=SelectSheetInput,
            ),
            StructuredTool.from_function(
                name="run_python",
                func=run_python,
                description=(
                    "Execute Python code for survey analysis. Use pandas operations and set variable `result` "
                    "for machine-readable output. If chart is requested, create matplotlib/seaborn charts with "
                    "clear labels; backend automatically captures generated figures."
                ),
                args_schema=PythonCodeInput,
            ),
        ]

    def _extract_charts_from_messages(self, messages: list[Any]) -> list[dict[str, str]]:
        charts: list[dict[str, str]] = []

        for msg in messages:
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue

            try:
                parsed = json.loads(content)
            except Exception:
                continue

            if isinstance(parsed, dict) and isinstance(parsed.get("charts"), list):
                for item in parsed["charts"]:
                    if not isinstance(item, dict):
                        continue
                    image_base64 = item.get("image_base64")
                    if not image_base64:
                        continue
                    charts.append(
                        {
                            "mime_type": str(item.get("mime_type", "image/png")),
                            "image_base64": str(image_base64),
                            "title": str(item.get("title", "")),
                        }
                    )

        if len(charts) > 6:
            charts = charts[-6:]

        logger.info("agent_charts_extracted count=%s", len(charts))
        return charts

    def _history_messages(self, session: SurveySession) -> list[tuple[str, str]]:
        history = session.chat_history[-self.max_history_messages :]
        messages: list[tuple[str, str]] = []

        for item in history:
            role = item.get("role", "").strip().lower()
            content = item.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(("human", content))
            elif role == "assistant":
                messages.append(("assistant", content))

        logger.info("history_context_loaded messages=%s", len(messages))
        return messages

    def answer(self, session: SurveySession, question: str) -> dict[str, Any]:
        logger.info(
            "agent_request_start session_id=%s active_sheet=%s question=%s",
            session.session_id,
            session.active_sheet,
            question,
        )
        tools = self._build_tools(session)
        agent = create_react_agent(self.llm, tools)

        system_prompt = (
            "You are a senior survey analyst agent. You answer user questions using the provided Excel data. "
            "Always inspect schema first when needed, and use run_python for calculations. "
            "If data quality problems exist, call them out. Be precise and concise. "
            "When returning numeric results, include clear labels and short interpretation. "
            "If user asks for charts, generate actual matplotlib or seaborn charts via run_python and reference "
            "the visual findings briefly in the final answer. "
            "Use prior conversation context to answer follow-up questions consistently."
        )

        history_messages = self._history_messages(session)
        request_messages: list[tuple[str, str]] = [("system", system_prompt)]
        request_messages.extend(history_messages)
        request_messages.append(
            (
                "human",
                (
                    f"Question: {question}\n"
                    f"Context: Active sheet is '{session.active_sheet}'. "
                    f"Available sheets: {', '.join(session.sheet_names)}"
                ),
            )
        )

        graph_response = agent.invoke(
            {
                "messages": request_messages
            }
        )
        messages = graph_response.get("messages", [])
        self._log_agent_messages(messages)
        if not messages:
            logger.warning("agent_request_end status=no_messages session_id=%s", session.session_id)
            return {"answer": "I could not produce an analysis response.", "charts": []}

        charts = self._extract_charts_from_messages(messages)
        final_answer = str(messages[-1].content)
        logger.info("agent_request_end status=ok session_id=%s", session.session_id)
        logger.info("agent_final_answer=%s", _truncate(final_answer, max_len=2000))
        return {"answer": final_answer, "charts": charts}
