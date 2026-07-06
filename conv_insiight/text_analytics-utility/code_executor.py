from __future__ import annotations

import ast
import base64
import contextlib
import io
import logging
import multiprocessing
from typing import Any

import pandas as pd

try:
    import resource as _resource       # Linux only
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False


logger = logging.getLogger("survey_agent.executor")

# ─── Static screen (read the code before running it) ────────────────────────
_ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "re", "json",
                    "matplotlib", "seaborn", "io"}
_BLOCKED_NAMES = {"os", "sys", "open", "__import__", "eval", "exec", "compile",
                  "globals", "locals", "__builtins__", "breakpoint", "input",
                  "socket", "subprocess", "requests", "urllib", "shutil", "pathlib"}


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                self.errors.append(f"Import not allowed: {a.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] not in _ALLOWED_IMPORTS:
            self.errors.append(f"Import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES:
            self.errors.append(f"Name not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append(f"Dunder attribute not allowed: {node.attr}")
        self.generic_visit(node)


def _check(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]
    c = _Checker()
    c.visit(tree)
    return c.errors


# ─── Isolated execution (separate process + resource caps) ──────────────────
def _worker(code: str, df: pd.DataFrame, conn) -> None:
    if _HAS_RESOURCE:
        # CPU-seconds cap bounds runaway loops; a wall-clock timeout in the
        # parent is the hard stop.
        try:
            _resource.setrlimit(_resource.RLIMIT_CPU, (15, 15))
        except (ValueError, OSError):
            pass

    import numpy as np
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception:
        sns = None

    scope: dict[str, Any] = {"df": df, "pd": pd, "np": np, "math": math,
                             "plt": plt, "sns": sns, "io": io, "result": None}
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(code, scope, scope)  # noqa: S102 — screened by AST + isolated process
    except Exception as exc:
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "charts": [], "stdout": out.getvalue()})
        conn.close()
        return

    charts = []
    for num in plt.get_fignums()[:4]:
        fig = plt.figure(num)
        if not any(ax.has_data() for ax in fig.axes):
            continue
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
        buf.seek(0)
        title = fig.axes[0].get_title() if fig.axes else ""
        charts.append({"mime_type": "image/png",
                       "image_base64": base64.b64encode(buf.read()).decode("utf-8"),
                       "title": title})
        plt.close(fig)

    res = scope.get("result")
    if hasattr(res, "to_dict"):
        try:
            res = res.to_dict()
        except Exception:
            res = str(res)
    conn.send({"ok": True, "result": res, "charts": charts, "stdout": out.getvalue()})
    conn.close()


def run_python_analysis_code(code: str, df: pd.DataFrame, timeout: int = 20) -> dict[str, Any]:
    """Run LLM-written analysis code against the session's combined (in-memory) dataset.

    Screened by AST first, then executed in a separate process with CPU/time
    limits. Exposes: df (all files combined), pd, np, plt, sns.
    """
    errs = _check(code)
    if errs:
        return {"ok": False, "error": "Code not allowed: " + "; ".join(errs), "charts": [], "stdout": ""}

    parent, child = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker, args=(code, df, child))
    p.start()
    child.close()
    if parent.poll(timeout):
        data = parent.recv()
    else:
        p.terminate()
        p.join(5)
        data = {"ok": False, "error": f"Execution timed out after {timeout}s", "charts": [], "stdout": ""}
    p.join()
    parent.close()
    return data
