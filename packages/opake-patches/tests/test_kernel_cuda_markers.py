"""Check CUDA markers on kernel test modules."""

from __future__ import annotations

import ast
from pathlib import Path


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _module_pytestmark(module: ast.Module) -> ast.expr | None:
    for statement in reversed(module.body):
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in statement.targets
        ):
            return statement.value
    return None


def _is_cuda_unavailable_skip(node: ast.AST) -> bool:
    if (
        not isinstance(node, ast.Call)
        or _qualified_name(node.func) != "pytest.mark.skipif"
        or not node.args
    ):
        return False

    condition = node.args[0]
    return (
        isinstance(condition, ast.UnaryOp)
        and isinstance(condition.op, ast.Not)
        and isinstance(condition.operand, ast.Call)
        and _qualified_name(condition.operand.func) == "torch.cuda.is_available"
    )


def test_cuda_skipped_kernel_modules_have_cuda_marker() -> None:
    kernel_tests = sorted((Path(__file__).parent / "kernels").glob("test_*.py"))
    assert kernel_tests, "No kernel test modules found"

    missing = []
    for path in kernel_tests:
        module = ast.parse(path.read_text(encoding="utf-8"))
        marker = _module_pytestmark(module)
        if marker is None:
            continue

        has_cuda_skip = any(
            _is_cuda_unavailable_skip(node) for node in ast.walk(marker)
        )
        has_cuda_marker = any(
            _qualified_name(node) == "pytest.mark.cuda" for node in ast.walk(marker)
        )
        if has_cuda_skip and not has_cuda_marker:
            missing.append(path.name)

    assert not missing, (
        "CUDA-skipped kernel modules lack pytest.mark.cuda: " + ", ".join(missing)
    )
