"""Regression guard for syntax-oriented public API documentation."""

import ast
from pathlib import Path


def test_modules_and_public_top_level_apis_have_docstrings():
    """Require documentation where notebook users are likely to import an object."""
    root = Path(__file__).parents[1] / "src" / "olp_ai_26"
    missing = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root)
        if ast.get_docstring(tree) is None:
            missing.append(f"{relative}: module")
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and ast.get_docstring(node) is None:
                    missing.append(f"{relative}:{node.lineno} {node.name}")
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not method.name.startswith("_") and ast.get_docstring(method) is None:
                            missing.append(f"{relative}:{method.lineno} {node.name}.{method.name}")
    assert not missing, "Missing public docstrings:\n" + "\n".join(missing)
