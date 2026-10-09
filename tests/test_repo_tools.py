"""Read-only repo tools under the conditions a real model repo creates.

Computational-biology repos routinely carry large *text* files - alignments,
structures, sequence dumps - which the binary-extension list does not cover.
Before these guards, one of them could exhaust memory or the context budget.
"""
from pathlib import Path

import pytest

from pipeline.tools.repo_tools import make_repo_tools


@pytest.fixture
def tools(tmp_path: Path):
    return {t.name: t for t in make_repo_tools(tmp_path, writable=False)}


def test_read_file_pages_without_loading_the_whole_file(tmp_path, tools):
    (tmp_path / "big.a3m").write_text("".join(f"line{i}\n" for i in range(100_000)))
    out = tools["read_file"].invoke({"path": "big.a3m", "start_line": 5, "end_line": 7})
    assert "5\tline4" in out and "7\tline6" in out
    assert "line7" not in out
    assert "continue from line 8" in out


def test_read_file_past_end_of_file_is_explicit(tmp_path, tools):
    (tmp_path / "short.txt").write_text("a\nb\n")
    assert "no lines in range" in tools["read_file"].invoke(
        {"path": "short.txt", "start_line": 50, "end_line": 60})


def test_whole_short_file_has_no_continuation_note(tmp_path, tools):
    (tmp_path / "s.py").write_text("a = 1\nb = 2\n")
    out = tools["read_file"].invoke({"path": "s.py"})
    assert "1\ta = 1" in out and "continue from" not in out


def test_single_enormous_line_is_clipped(tmp_path, tools):
    (tmp_path / "one.fasta").write_text(">x\n" + "M" * 500_000 + "\n")
    out = tools["read_file"].invoke({"path": "one.fasta"})
    assert len(out) < 10_000
    assert "500000 chars" in out


def test_binary_files_are_not_read(tmp_path, tools):
    (tmp_path / "w.pt").write_bytes(b"\x00\x01\x02")
    assert "binary file" in tools["read_file"].invoke({"path": "w.pt"})


def test_search_skips_oversized_data_files(tmp_path, tools):
    (tmp_path / "db.a3m").write_text("NEEDLE\n" + "x" * 3_000_000)
    (tmp_path / "code.py").write_text("NEEDLE here\n")
    out = tools["search"].invoke({"regex": "NEEDLE"})
    assert "code.py:1" in out
    assert "db.a3m" not in out
    assert "skipped" in out


def test_search_reports_a_bad_regex_instead_of_crashing(tmp_path, tools):
    (tmp_path / "a.py").write_text("x\n")
    assert "invalid regex" in tools["search"].invoke({"regex": "(unclosed"})


def test_search_finds_nothing_cleanly(tmp_path, tools):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert tools["search"].invoke({"regex": "zzz"}).startswith("(no matches)")


def test_path_traversal_still_blocked(tmp_path, tools):
    with pytest.raises(Exception):
        tools["read_file"].invoke({"path": "../../etc/passwd"})


def test_symlink_escaping_the_repo_is_blocked(tmp_path, tools):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(Exception):
        tools["read_file"].invoke({"path": "link.txt"})
