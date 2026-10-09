"""File tools scoped to one workspace directory (the cloned repo)."""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".tox"}
BINARY_EXT = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".h5", ".ckpt", ".pkl", ".npy", ".npz",
              ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz", ".so", ".dylib", ".pyc", ".parquet"}

# Guards for repositories carrying large *text* data - alignments, structures,
# sequence dumps - which BINARY_EXT does not cover and which are common in
# computational-biology repos.
SEARCH_MAX_FILE_BYTES = 2_000_000
MAX_LINE_CHARS = 2_000


def _clip_line(line: str) -> str:
    """One-line FASTA/JSON files exist; a single line must not blow the context budget."""
    return line if len(line) <= MAX_LINE_CHARS else f"{line[:MAX_LINE_CHARS]}... [{len(line)} chars]"


def make_repo_tools(workspace: Path, *, writable: bool) -> list[BaseTool]:
    root = workspace.resolve()

    def _resolve(rel: str) -> Path:
        p = (root / rel).resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"path '{rel}' is outside the repository")
        return p

    def list_dir(path: str = ".", max_depth: int = 2) -> str:
        """List files and folders under `path` (relative to repo root) up to `max_depth` levels, with sizes."""
        base = _resolve(path)
        lines: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            depth = len(Path(dirpath).relative_to(base).parts)
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            if depth >= max_depth:
                dirnames[:] = []
            for fn in sorted(filenames):
                fp = Path(dirpath) / fn
                lines.append(f"{fp.relative_to(root)}  ({fp.stat().st_size} B)")
            if len(lines) > 800:
                lines.append("... (truncated; narrow the path)")
                break
        return "\n".join(lines) or "(empty)"

    def find_files(pattern: str) -> str:
        """Find files whose path matches a glob pattern, e.g. '**/*.py', '*requirements*.txt', 'tests/**'."""
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                rel = str((Path(dirpath) / fn).relative_to(root))
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fn, pattern):
                    hits.append(rel)
        return "\n".join(sorted(hits)[:500]) or "(no matches)"

    def read_file(path: str, start_line: int = 1, end_line: int = 400) -> str:
        """Read a text file with line numbers. Use start_line/end_line to page through large files."""
        p = _resolve(path)
        if p.suffix.lower() in BINARY_EXT:
            return f"(binary file, {p.stat().st_size} bytes - not shown)"
        # Streamed, not read_text(): a model repo can carry a multi-GB FASTA, PDB or CSV,
        # and slurping one to return 400 lines would take the orchestrator down with it.
        start, out, truncated = max(start_line, 1), [], False
        with p.open(errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                if n < start:
                    continue
                if n > end_line:
                    truncated = True
                    break
                out.append(f"{n}\t{_clip_line(line.rstrip(chr(10)))}")
        if not out:
            return f"(no lines in range {start}-{end_line}; file has fewer lines)"
        more = f"\n... (more lines follow; continue from line {end_line + 1})" if truncated else ""
        return "\n".join(out) + more

    def search(regex: str, glob: str = "*", max_results: int = 200) -> str:
        """Search file contents with a Python regex. `glob` filters file names (e.g. '*.py')."""
        try:
            rx = re.compile(regex)
        except re.error as e:  # a bad pattern is the model's mistake to fix, not a crash
            return f"ERROR: invalid regex {regex!r}: {e}"
        out, skipped = [], 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fnmatch.fnmatch(fn, glob) or Path(fn).suffix.lower() in BINARY_EXT:
                    continue
                fp = Path(dirpath) / fn
                try:
                    # Data files (alignments, structures, CSV dumps) are not where the
                    # inference path lives, and scanning them dominates the runtime.
                    if fp.stat().st_size > SEARCH_MAX_FILE_BYTES:
                        skipped += 1
                        continue
                    with fp.open(errors="replace") as fh:
                        for n, line in enumerate(fh, 1):
                            if rx.search(line):
                                out.append(f"{fp.relative_to(root)}:{n}: {line.strip()[:200]}")
                                if len(out) >= max_results:
                                    return "\n".join(out) + "\n... (max results reached)"
                except OSError:
                    continue
        note = f"\n... ({skipped} file(s) over {SEARCH_MAX_FILE_BYTES // 1_000_000} MB skipped)" if skipped else ""
        return ("\n".join(out) or "(no matches)") + note

    tools = [
        StructuredTool.from_function(list_dir),
        StructuredTool.from_function(find_files),
        StructuredTool.from_function(read_file),
        StructuredTool.from_function(search),
    ]
    if not writable:
        return tools

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in the repo with `content`."""
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {path} ({len(content)} chars)"

    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace one exact occurrence of `old_text` with `new_text` in a file."""
        p = _resolve(path)
        src = p.read_text()
        count = src.count(old_text)
        if count != 1:
            return f"ERROR: old_text found {count} times; it must match exactly once. Read the file and retry."
        p.write_text(src.replace(old_text, new_text))
        return f"edited {path}"

    return tools + [StructuredTool.from_function(write_file), StructuredTool.from_function(edit_file)]
