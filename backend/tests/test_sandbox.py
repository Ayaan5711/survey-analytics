from __future__ import annotations
import pytest
import pandas as pd
import io
from pathlib import Path
from app.sandbox.ast_check import check_ast
from app.sandbox.runner import run_code, SandboxResult


@pytest.fixture
def parquet_file(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)
    return str(path)


# AST check tests
def test_ast_allows_pandas_import():
    errors = check_ast("import pandas as pd\ndf.head()")
    assert errors == []


def test_ast_allows_numpy_matplotlib():
    errors = check_ast("import numpy as np\nimport matplotlib.pyplot as plt")
    assert errors == []


def test_ast_blocks_os_import():
    errors = check_ast("import os")
    assert any("os" in e for e in errors)


def test_ast_blocks_sys_import():
    errors = check_ast("import sys")
    assert any("sys" in e for e in errors)


def test_ast_blocks_open_builtin():
    errors = check_ast("open('file.txt')")
    assert any("open" in e for e in errors)


def test_ast_blocks_dunder_attribute():
    errors = check_ast("df.__class__")
    assert any("__" in e for e in errors)


def test_ast_blocks_from_os_import():
    errors = check_ast("from os import path")
    assert any("os" in e for e in errors)


def test_ast_catches_syntax_error():
    errors = check_ast("def foo(:")
    assert any("Syntax" in e for e in errors)


# Runner tests
def test_run_code_basic(parquet_file):
    code = "result_summary = f'rows={len(df)}'"
    result = run_code(code, parquet_file)
    assert result.success
    assert result.summary == "rows=3"
    assert result.error is None


def test_run_code_produces_png(parquet_file):
    code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
fig, ax = plt.subplots()
ax.bar(df['B'], df['A'])
buf = io.BytesIO()
fig.savefig(buf, format='png')
plt.close()
buf.seek(0)
result_png = buf.read()
result_summary = 'bar chart done'
"""
    result = run_code(code, parquet_file)
    assert result.success, result.error
    assert isinstance(result.png_bytes, bytes)
    assert len(result.png_bytes) > 0


def test_run_code_catches_runtime_error(parquet_file):
    code = "x = 1 / 0"
    result = run_code(code, parquet_file)
    assert not result.success
    assert "ZeroDivision" in result.error


def test_run_code_timeout(parquet_file):
    code = "import time; time.sleep(60)"
    result = run_code(code, parquet_file, timeout=2)
    assert not result.success
    assert "timed out" in result.error.lower()


def test_ast_blocked_code_not_run(parquet_file):
    code = "import os; result_summary = os.getcwd()"
    result = run_code(code, parquet_file)
    assert not result.success
    assert "not allowed" in result.error.lower()
