from __future__ import annotations

import base64
import contextlib
import importlib
import io
import logging
import math
from builtins import __import__ as py_import
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "print": print,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "isinstance": isinstance,
    "hasattr": hasattr,
}


SAFE_IMPORTS = {
    "pandas",
    "numpy",
    "math",
    "statistics",
    "re",
    "json",
    "matplotlib",
    "seaborn",
}


def restricted_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    root = name.split(".", 1)[0]
    if root not in SAFE_IMPORTS:
        raise ImportError(f"Import of '{name}' is blocked")
    return py_import(name, globals, locals, fromlist, level)


ALLOWED_BUILTINS["__import__"] = restricted_import


logger = logging.getLogger("survey_agent.executor")


def _truncate(value: Any, max_len: int = 1500) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


def _collect_chart_images(max_charts: int = 4) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    figure_numbers = list(plt.get_fignums())[:max_charts]

    for figure_number in figure_numbers:
        figure = plt.figure(figure_number)

        # Ignore placeholder/empty figures so UI does not show blank charts.
        has_chart_data = any(axis.has_data() for axis in figure.axes)
        if not has_chart_data:
            logger.info("python_exec_chart_skipped_empty figure_number=%s", figure_number)
            plt.close(figure)
            continue

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=130)
        buffer.seek(0)

        title = ""
        if figure.axes:
            title = figure.axes[0].get_title() or ""

        charts.append(
            {
                "mime_type": "image/png",
                "image_base64": base64.b64encode(buffer.read()).decode("utf-8"),
                "title": title,
            }
        )

        buffer.close()
        plt.close(figure)

    return charts


def _load_seaborn_if_available():
    try:
        return importlib.import_module("seaborn")
    except Exception:
        return None


def run_python_analysis_code(code: str, dataframes: dict[str, pd.DataFrame], active_sheet: str) -> dict[str, Any]:
    """
    Executes analysis code in a constrained namespace.

    Exposed variables:
    - df: active sheet dataframe
    - dataframes: all workbook sheets by name
    - pd: pandas module

    To return structured content, assign to variable `result` in user code.
    """
    output_buffer = io.StringIO()
    exec_scope: dict[str, Any] = {
        "pd": pd,
        "math": math,
        "plt": plt,
        "sns": _load_seaborn_if_available(),
        "dataframes": dataframes,
        "df": dataframes[active_sheet],
        "result": None,
        "__builtins__": ALLOWED_BUILTINS,
    }

    plt.close("all")

    logger.info(
        "python_exec_start active_sheet=%s rows=%s cols=%s",
        active_sheet,
        dataframes[active_sheet].shape[0],
        dataframes[active_sheet].shape[1],
    )
    logger.info("python_exec_code=\n%s", code)

    try:
        with contextlib.redirect_stdout(output_buffer):
            # Use one shared scope so helper functions can resolve names like `pd`.
            exec(code, exec_scope, exec_scope)
    except Exception as exc:
        plt.close("all")
        error_payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": output_buffer.getvalue(),
            "charts": [],
        }
        logger.exception("python_exec_error payload=%s", _truncate(error_payload))
        return error_payload

    result_obj = exec_scope.get("result")
    if hasattr(result_obj, "to_dict"):
        try:
            result_obj = result_obj.to_dict()
        except Exception:
            result_obj = str(result_obj)

    charts = _collect_chart_images(max_charts=4)

    success_payload = {
        "ok": True,
        "stdout": output_buffer.getvalue(),
        "result": result_obj,
        "charts": charts,
    }
    logger.info("python_exec_charts_generated count=%s", len(charts))
    logger.info("python_exec_success payload=%s", _truncate(success_payload))
    return success_payload
