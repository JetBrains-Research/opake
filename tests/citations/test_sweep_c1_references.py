"""Regression guard for Sweep C1 bibliographic citations (#1039).

Generates a per-file checklist from docstrings and comments: every scholarly
reference in the three audited sites must pair with a stable identifier (arXiv
id or DOI) and must not resurrect the known-wrong records from OPQ-371/372/375.
"""

from __future__ import annotations

import ast
import re
import tokenize
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.IGNORECASE)
DOI_RE = re.compile(r"doi\.org/([^\s\])>\"']+)", re.IGNORECASE)

# Sites audited in JetBrains-Research/opake#1039 (Sweep C1).
SWEEP_C1_FILES = (
    REPO_ROOT / "packages/opake-optimizers/src/opake/api/optimizers/_schedule_free.py",
    REPO_ROOT
    / "packages/opake-alignment/src/opake/api/alignment/dpo/loss/_discopop.py",
    REPO_ROOT / "packages/opake-alignment/src/opake/api/alignment/logprob/_sequence.py",
    REPO_ROOT / "packages/opake-dpftrl/src/opake/api/dpftrl/sampling/_balls_in_bins.py",
)

FORBIDDEN_SNIPPETS = (
    "Yaida",
    "Using Self-Supervised Feedback",
    '"Privacy Amplification for Matrix Mechanisms"',  # wrong title (no "Near Exact")
)

EXPECTED_FILE_IDENTIFIERS: dict[Path, tuple[str, ...]] = {
    SWEEP_C1_FILES[0]: ("2405.15682",),
    SWEEP_C1_FILES[1]: ("2406.08414",),
    SWEEP_C1_FILES[2]: ("2409.06411",),
    SWEEP_C1_FILES[3]: ("2410.06266",),
}


@dataclass(frozen=True)
class CitationChecklistEntry:
    """One docstring/comment reference paired with stable identifiers."""

    path: Path
    kind: str
    location: str
    text: str
    arxiv_ids: tuple[str, ...]
    dois: tuple[str, ...]


def _docstring_nodes(tree: ast.AST) -> list[tuple[str, str]]:
    nodes: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        if isinstance(node, ast.Module):
            label = "module"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            label = f"{type(node).__name__}:{node.name}"
        else:
            label = f"ClassDef:{node.name}"
        nodes.append((label, doc))
    return nodes


def _comment_nodes(source: str) -> list[tuple[str, str]]:
    comments: list[tuple[str, str]] = []
    for tok in tokenize.tokenize(BytesIO(source.encode()).readline):
        if tok.type == tokenize.COMMENT:
            comments.append(
                (f"comment:L{tok.start[0]}", tok.string.lstrip("# ").strip())
            )
    return comments


def _stable_ids(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    arxiv = tuple(sorted({m.group(1) for m in ARXIV_ID_RE.finditer(text)}))
    dois = tuple(sorted({m.group(1) for m in DOI_RE.finditer(text)}))
    return arxiv, dois


def citation_checklist(path: Path) -> list[CitationChecklistEntry]:
    """Build a per-file checklist from module docstrings and comments."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    entries: list[CitationChecklistEntry] = []
    for kind, text in _docstring_nodes(tree) + _comment_nodes(source):
        arxiv_ids, dois = _stable_ids(text)
        if not arxiv_ids and not dois:
            continue
        entries.append(
            CitationChecklistEntry(
                path=path,
                kind=kind,
                location=str(path.relative_to(REPO_ROOT)),
                text=text,
                arxiv_ids=arxiv_ids,
                dois=dois,
            )
        )
    return entries


def _file_doc_comment_text(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    chunks = [text for kind, text in _docstring_nodes(tree) if kind == "module"]
    chunks.extend(text for _, text in _comment_nodes(source))
    return "\n".join(chunks)


@pytest.mark.parametrize("path", SWEEP_C1_FILES, ids=lambda p: p.name)
def test_sweep_c1_file_has_expected_identifier(path: Path) -> None:
    """Each audited file exposes the canonical arXiv id for its primary record."""
    text = _file_doc_comment_text(path)
    for snippet in FORBIDDEN_SNIPPETS:
        assert snippet not in text, (
            f"{path.name} still contains forbidden snippet: {snippet!r}"
        )

    expected = EXPECTED_FILE_IDENTIFIERS[path]
    found = sorted({m.group(1) for m in ARXIV_ID_RE.finditer(text)})
    for arxiv_id in expected:
        assert arxiv_id in found, (
            f"{path.name}: expected arXiv:{arxiv_id} in docstrings/comments, "
            f"found {found or 'none'}"
        )


@pytest.mark.parametrize("path", SWEEP_C1_FILES, ids=lambda p: p.name)
def test_sweep_c1_checklist_entries_have_identifiers(path: Path) -> None:
    """Every extracted reference block in the file carries a stable identifier."""
    checklist = citation_checklist(path)
    assert checklist, f"{path.name}: no docstring/comment references with identifiers"
    for entry in checklist:
        assert entry.arxiv_ids or entry.dois, (
            f"{entry.location} [{entry.kind}] lacks arXiv/DOI identifier"
        )


def test_schedule_free_authors_exclude_yaida() -> None:
    path = SWEEP_C1_FILES[0]
    text = _file_doc_comment_text(path)
    assert "Yang" in text
    assert "Cutkosky" in text
    assert "The Road Less Scheduled" in text


def test_discopop_cites_lu_et_al_record() -> None:
    path = SWEEP_C1_FILES[1]
    text = _file_doc_comment_text(path)
    assert "Lu" in text
    assert "Discovering Preference Optimization Algorithms" in text
    assert "2406.08414" in text


def test_balls_in_bins_cites_near_exact_amplification_paper() -> None:
    path = SWEEP_C1_FILES[3]
    text = re.sub(r"\s+", " ", _file_doc_comment_text(path))
    assert "Near Exact Privacy Amplification for Matrix Mechanisms" in text
    assert "2410.06266" in text
    assert "Definition 3.1" in text
    assert "Lemma 3.2" in text


def test_sequence_module_docstring_covers_ld_alpha_and_fused_limit() -> None:
    path = SWEEP_C1_FILES[2]
    module_doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert module_doc is not None
    assert "ld_alpha" in module_doc
    assert "2409.06411" in module_doc
    assert "fused_sequence_logp" in module_doc
    assert "does not support" in module_doc.lower()
