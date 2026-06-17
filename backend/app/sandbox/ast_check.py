from __future__ import annotations
import ast

_ALLOWED_IMPORTS = {"pandas", "pd", "numpy", "np", "matplotlib", "plt", "seaborn", "plotly", "math", "time"}
_BLOCKED_NAMES = {"os", "sys", "open", "__import__", "eval", "exec", "compile",
                  "globals", "locals", "__builtins__", "breakpoint", "input"}


class _ASTChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base = alias.name.split(".")[0]
            if base not in _ALLOWED_IMPORTS:
                self.errors.append(f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base = node.module.split(".")[0]
            if base not in _ALLOWED_IMPORTS:
                self.errors.append(f"Import not allowed: from {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES:
            self.errors.append(f"Name not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append(f"Dunder attribute access not allowed: {node.attr}")
        self.generic_visit(node)


def check_ast(code: str) -> list[str]:
    """Return list of violation strings; empty list means code is safe to run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]
    checker = _ASTChecker()
    checker.visit(tree)
    return checker.errors
