from __future__ import annotations
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from app.data.profiler import DatasetProfile
from app.llm.client import llm
from app.llm.prompts import chat_system_prompt, tool_definitions, synthesis_prompt, code_gen_prompt
from app.sandbox.runner import run_code, SandboxResult
from app.tools.registry import dispatch_tool, ToolResult

_MAX_STEPS = 4


@dataclass
class ChatResponse:
    role: str
    content: str
    chart: dict | None          # {png_b64, title} or {plotly_json, title}
    generated_code: str | None
    follow_ups: list[str]
    caveats: list[str]
    tool_calls_made: list[str]
    table: list | dict | None = None     # deterministic facts from the last tool
    table_title: str | None = None


class ChatAgent:
    async def run(
        self,
        profile: DatasetProfile,
        parquet_path: str,
        message: str,
        history: list[dict],
        compare_diff: str | None = None,
    ) -> ChatResponse:
        system = chat_system_prompt(profile)
        if compare_diff:
            system += (
                "\n\nA comparison dataset is active. Differences between the base and "
                "comparison datasets:\n" + compare_diff +
                "\nWhen the user asks what changed, ground your answer in these deltas."
            )
        messages: list[dict] = [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": message},
        ]

        df = pd.read_parquet(parquet_path)
        tool_calls_made: list[str] = []
        tool_summaries: list[str] = []
        last_chart: dict | None = None
        last_code: str | None = None
        last_table: list | dict | None = None
        last_table_title: str | None = None
        caveats: list[str] = []

        for _step in range(_MAX_STEPS):
            response = await llm._client.chat.completions.create(
                model=llm.deployment_fallback,
                messages=messages,
                tools=tool_definitions(),
                tool_choice="auto",
                temperature=0.2,
                max_tokens=800,
            )
            choice = response.choices[0]

            if choice.finish_reason == "stop":
                break

            if choice.finish_reason == "tool_calls":
                msg_dict = choice.message.model_dump(exclude_none=True)
                messages.append(msg_dict)

                for tc in choice.message.tool_calls:
                    name = tc.function.name
                    params = json.loads(tc.function.arguments)
                    tool_calls_made.append(name)

                    result = await self._execute(name, params, df, parquet_path, profile, message)
                    if result.chart:
                        last_chart = result.chart
                    if result.caveat:
                        caveats.append(result.caveat)
                    if result.code:
                        last_code = result.code
                    if result.table is not None:
                        last_table = result.table
                        last_table_title = result.table_title
                    tool_summaries.append(result.summary)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.summary[:2000],
                    })

        # Synthesis call
        synth_msg = synthesis_prompt(message, tool_summaries)
        raw = await llm.chat_completion(
            messages=[{"role": "user", "content": synth_msg}],
            use_fallback=False,
            max_tokens=400,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(raw)
            narrative = parsed.get("narrative", raw)
            follow_ups = parsed.get("follow_ups", [])[:3]
        except (json.JSONDecodeError, AttributeError):
            narrative = raw
            follow_ups = []

        return ChatResponse(
            role="assistant",
            content=narrative,
            chart=last_chart,
            generated_code=last_code,
            follow_ups=follow_ups,
            caveats=caveats,
            tool_calls_made=tool_calls_made,
            table=last_table,
            table_title=last_table_title,
        )

    @dataclass
    class _ExecResult:
        summary: str
        chart: dict | None = None
        caveat: str | None = None
        code: str | None = None
        table: list | dict | None = None
        table_title: str | None = None

    async def _execute(
        self,
        name: str,
        params: dict,
        df: pd.DataFrame,
        parquet_path: str,
        profile: DatasetProfile,
        message: str,
    ) -> "ChatAgent._ExecResult":
        if name == "generate_code":
            code = params.get("code", "")
            result = run_code(code, parquet_path)
            chart = self._chart_from_sandbox(result)
            # Fix-and-retry once with the primary model if the first attempt
            # failed or produced no chart at all.
            if not result.success or chart is None:
                error = result.error or "No chart was produced (result_png was never set)."
                fixed = await llm.chat_completion(
                    messages=[{"role": "user",
                               "content": code_gen_prompt(message, profile, error)}],
                    use_fallback=False,
                    max_tokens=900,
                    temperature=0.1,
                )
                fixed_code = _strip_code_fences(fixed)
                retry = run_code(fixed_code, parquet_path)
                retry_chart = self._chart_from_sandbox(retry)
                if retry.success and retry_chart is not None:
                    return self._ExecResult(
                        summary=retry.summary or "Custom chart generated.",
                        chart=retry_chart, code=fixed_code)
                # Both attempts failed — surface the latest code + a clear note.
                return self._ExecResult(
                    summary=f"Could not render this chart automatically ({retry.error or error}).",
                    code=fixed_code or code)
            return self._ExecResult(summary=result.summary or "Custom chart generated.",
                                    chart=chart, code=code)

        tr: ToolResult = dispatch_tool(df, name, params)
        chart = None
        if tr.png_bytes:
            chart = {"png_b64": base64.b64encode(tr.png_bytes).decode(), "title": tr.summary}
        table_str = json.dumps(tr.table)[:500] if tr.table else ""
        return self._ExecResult(
            summary=f"{tr.summary}. Data: {table_str}",
            chart=chart,
            caveat=tr.caveat,
            table=tr.table,
            table_title=tr.summary,
        )

    @staticmethod
    def _chart_from_sandbox(result: SandboxResult) -> dict | None:
        if not result.success:
            return None
        if result.png_bytes:
            return {"png_b64": base64.b64encode(result.png_bytes).decode(), "title": "Custom chart"}
        if result.plotly_json:
            return {"plotly_json": result.plotly_json, "title": "Custom chart"}
        return None


def _strip_code_fences(text: str) -> str:
    """Remove ```python ... ``` fences the model may add despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()
