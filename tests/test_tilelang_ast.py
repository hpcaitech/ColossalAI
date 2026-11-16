"""TileLang AST Checker & Tests.

Enforces prohibition of `from __future__ import annotations` in TileLang kernel files.
Background:
TileLang's `@T.prim_func` inspects parameter type annotations as TVM objects (kTVMObjectHandle).
PEP 563 postponed annotations (`from __future__ import annotations`) convert annotations
into string literals, causing TVM's C++ AST parser to crash with:
`InternalError: Check failed: type_code_ == kTVMObjectHandle (11 vs. 8) : expected Object but got str`
"""

import ast
from pathlib import Path
from typing import Optional, Union

import pytest


class TileLangASTViolationError(ValueError):
    """Raised when a TileLang kernel file violates AST rules."""


class TileLangASTScanner(ast.NodeVisitor):
    """AST scanner inspecting Python files for TileLang usage and illegal future annotations."""

    def __init__(self, filename: Optional[str] = None):
        self.filename = filename or "<string>"
        self.has_future_annotations = False
        self.uses_tilelang = False

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    self.has_future_annotations = True
        elif node.module and "tilelang" in node.module:
            self.uses_tilelang = True
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if "tilelang" in alias.name:
                self.uses_tilelang = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for dec in node.decorator_list:
            dec_id = ""
            if isinstance(dec, ast.Name):
                dec_id = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_id = f"{getattr(dec.value, 'id', '')}.{dec.attr}"
            if "prim_func" in dec_id or "tilelang" in dec_id:
                self.uses_tilelang = True
        self.generic_visit(node)


def check_tilelang_source(source_code: str, filename: Optional[str] = None) -> None:
    """Scan Python source code and assert it satisfies TileLang AST constraints."""
    tree = ast.parse(source_code, filename=filename or "<string>")
    scanner = TileLangASTScanner(filename=filename)
    scanner.visit(tree)

    if scanner.uses_tilelang and scanner.has_future_annotations:
        raise TileLangASTViolationError(
            f"TileLang AST violation in {scanner.filename}: 'from __future__ import annotations' is strictly "
            f"prohibited in TileLang kernels. Postponing type evaluation converts annotations into strings, "
            f"triggering TVM parser failure: Check failed: type_code_ == kTVMObjectHandle (11 vs. 8) : "
            f"expected Object but got str."
        )


def check_tilelang_file(file_path: Union[str, Path]) -> None:
    """Read and validate a Python file on disk."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    content = p.read_text(encoding="utf-8")
    check_tilelang_source(content, filename=str(p))


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_valid_tilelang_kernel_source_passes():
    valid_source = """
import tilelang as T

@T.prim_func
def vec_add(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    n: T.int32,
):
    pass
"""
    # Should not raise
    check_tilelang_source(valid_source)


def test_tilelang_kernel_with_future_annotations_fails():
    invalid_source = """
from __future__ import annotations
import tilelang as T

@T.prim_func
def vec_add(
    a: T.handle,
    b: T.handle,
    c: T.handle,
):
    pass
"""
    with pytest.raises(TileLangASTViolationError, match="kTVMObjectHandle"):
        check_tilelang_source(invalid_source)


def test_non_tilelang_module_with_future_annotations_passes():
    normal_source = """
from __future__ import annotations
from typing import List, Optional

def standard_helper(x: List[int]) -> Optional[int]:
    return x[0] if x else None
"""
    # Normal Python modules without tilelang or prim_func should pass
    check_tilelang_source(normal_source)


def test_explicit_prim_func_decorator_triggers_check():
    source = """
from __future__ import annotations

@prim_func
def custom_kernel(x: int):
    pass
"""
    with pytest.raises(TileLangASTViolationError, match="strictly prohibited in TileLang"):
        check_tilelang_source(source)


def test_check_file_on_disk(tmp_path):
    good_file = tmp_path / "good_kernel.py"
    good_file.write_text(
        """
import tilelang as T

@T.prim_func
def gemm_kernel():
    pass
""",
        encoding="utf-8",
    )
    check_tilelang_file(good_file)

    bad_file = tmp_path / "bad_kernel.py"
    bad_file.write_text(
        """
from __future__ import annotations
import tilelang as T

@T.prim_func
def gemm_kernel():
    pass
""",
        encoding="utf-8",
    )
    with pytest.raises(TileLangASTViolationError):
        check_tilelang_file(bad_file)


def test_scan_repository_python_files_for_future_annotations():
    """Ensure no kernel in the repository violates the TileLang AST constraint."""
    repo_root = Path(__file__).resolve().parent.parent
    for py_file in repo_root.glob("examples/worldmodel/cosmos3/**/*.py"):
        try:
            check_tilelang_file(py_file)
        except TileLangASTViolationError as e:
            pytest.fail(f"Found TileLang AST violation in repository: {e}")
