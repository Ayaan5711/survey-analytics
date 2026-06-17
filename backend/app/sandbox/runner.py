from __future__ import annotations
import multiprocessing
import traceback
from dataclasses import dataclass

try:
    import resource as _resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

from app.sandbox.ast_check import check_ast


@dataclass
class SandboxResult:
    success: bool
    png_bytes: bytes | None
    plotly_json: str | None
    summary: str | None
    error: str | None


def _worker(code: str, parquet_path: str, conn) -> None:  # runs in subprocess
    if _HAS_RESOURCE:
        _resource.setrlimit(_resource.RLIMIT_CPU, (10, 10))
        try:
            _resource.setrlimit(_resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        except ValueError:
            pass

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        conn.send({"success": False, "png_bytes": None, "plotly_json": None,
                   "summary": None, "error": f"Failed to load data: {exc}"})
        conn.close()
        return

    local_ns: dict = {
        "df": df, "pd": pd, "np": np, "plt": plt, "io": io,
        "result_png": None, "result_plotly": None, "result_summary": None,
    }
    try:
        exec(code, local_ns)  # noqa: S102
        conn.send({
            "success": True,
            "png_bytes": local_ns.get("result_png"),
            "plotly_json": local_ns.get("result_plotly"),
            "summary": local_ns.get("result_summary"),
            "error": None,
        })
    except Exception as exc:
        conn.send({
            "success": False, "png_bytes": None, "plotly_json": None,
            "summary": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        })
    finally:
        conn.close()


def run_code(code: str, parquet_path: str, timeout: int = 15) -> SandboxResult:
    ast_errors = check_ast(code)
    if ast_errors:
        return SandboxResult(success=False, png_bytes=None, plotly_json=None,
                             summary=None, error="Code not allowed: " + "; ".join(ast_errors))

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker, args=(code, parquet_path, child_conn))
    p.start()
    child_conn.close()

    if parent_conn.poll(timeout):
        data = parent_conn.recv()
    else:
        p.terminate()
        p.join(timeout=5)
        data = {"success": False, "png_bytes": None, "plotly_json": None,
                "summary": None, "error": f"Sandbox timed out after {timeout}s"}

    p.join()
    parent_conn.close()
    return SandboxResult(**data)
