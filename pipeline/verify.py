"""Deterministic checks on an agent's analysis, run AFTER the LLM has answered.

Schema validation is not usefulness validation. Almost every field of
`RepoAnalysis` defaults to empty, so an analyst that exhausted its step budget
and returned two sentences still produces a valid object - and the graph would
walk on to provision a GPU on the strength of it.

These checks compare the analysis against two things the model cannot argue
with: the filesystem, and preflight's independent scan. Each discrepancy
becomes a risk line, and risk lines are rendered at the EC2 approval gate,
which is the last point where stopping is free.

Nothing here calls an LLM and nothing here raises: the pipeline should surface
a thin analysis to the human, not crash on one.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# A cited path with a glob, a line range or a shell fragment is not a claim
# about a specific file, so it is not checked for existence.
_NOT_A_PATH = re.compile(r"[*?\s<>|]")
# ...and neither is a bare symbol: `capabilities[].entrypoint` is legitimately
# a function name ("predict") as often as it is a file ("scripts/run.py").
_LOOKS_LIKE_PATH = re.compile(r"/|\.[A-Za-z0-9]{1,6}$")

# Where the analysis names files that downstream agents will act on.
_PATH_FIELDS = [
    ("inference.file", lambda a: [a.get("inference", {}).get("file", "")]),
    ("key_files", lambda a: [f.get("path", "") for f in a.get("key_files") or []]),
    ("functions", lambda a: [f.get("file", "") for f in a.get("functions") or []]),
    ("tests.paths", lambda a: list(a.get("tests", {}).get("paths") or [])),
    ("dependency_files", lambda a: list(a.get("dependency_files") or [])),
    ("scripts", lambda a: [s.get("path", "") for s in a.get("scripts") or []]),
    ("capabilities", lambda a: [c.get("entrypoint", "") for c in a.get("capabilities") or []]),
    # Only assets claimed to be IN the repo: one marked bundled=False is meant to
    # be downloaded, so its absence is correct rather than a mistake.
    ("assets", lambda a: [s.get("path", "") for s in a.get("assets") or [] if s.get("bundled", True)]),
]


def _cited_paths(analysis: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for field, get in _PATH_FIELDS:
        try:
            values = get(analysis)
        except (AttributeError, TypeError):
            continue
        for raw in values:
            if not isinstance(raw, str):
                continue
            p = raw.strip().lstrip("./").split(":")[0]
            if p and not _NOT_A_PATH.search(p) and _LOOKS_LIKE_PATH.search(p):
                out.append((field, p))
    return out


def missing_paths(analysis: dict, workspace: Path) -> list[tuple[str, str]]:
    """Cited paths that do not exist in the cloned repo, deduplicated."""
    seen: set[str] = set()
    missing: list[tuple[str, str]] = []
    for field, p in _cited_paths(analysis):
        if p in seen:
            continue
        seen.add(p)
        if not (workspace / p).exists():
            missing.append((field, p))
    return missing


def completeness_gaps(analysis: dict) -> list[str]:
    """Fields a later stage genuinely needs. Empty here means an agent is blocked later."""
    gaps: list[str] = []
    inf = analysis.get("inference") or {}
    tests = analysis.get("tests") or {}
    hw = analysis.get("hardware") or {}

    if not inf.get("file") and not inf.get("function_or_class"):
        gaps.append("no inference entrypoint identified - the adapter builder has nothing to call")
    if not inf.get("input_format") or not inf.get("output_format"):
        gaps.append("inference input/output format not recorded - the adapter cannot map the schema")
    if (tests.get("framework") or "").lower() not in ("none", "") and not tests.get("command"):
        gaps.append(f"tests use {tests['framework']} but no test command was recorded")
    if hw.get("gpu_required") and not hw.get("min_gpu_memory_gb"):
        gaps.append("GPU required but no VRAM estimate - the instance cannot be sized")
    # --- the orientation document's own requirements ----------------------
    if not analysis.get("capabilities"):
        gaps.append("no capabilities listed - a reader cannot tell what the repo can do")
    if not analysis.get("inputs") or not analysis.get("outputs"):
        gaps.append("inputs/outputs not described as data - the I/O contract is unusable")
    if not analysis.get("flow"):
        gaps.append("no process flow recorded - no diagram can be drawn")
    if not (analysis.get("metadata") or {}).get("purpose"):
        gaps.append("no stated purpose for the model")
    if not (analysis.get("licensing") or {}).get("weights"):
        gaps.append("weights licence not established - serving may not be permitted")
    return gaps


def contradictions(analysis: dict, pf: Any) -> list[str]:
    """Places the agent disagrees with the deterministic scan. The scan read the files."""
    out: list[str] = []
    hw = analysis.get("hardware") or {}
    tests = analysis.get("tests") or {}
    weights = analysis.get("weights") or {}

    if pf.cuda_pins and not hw.get("gpu_required"):
        out.append(f"scan found CUDA/accelerator pins ({', '.join(sorted(pf.cuda_pins))}) "
                   "but the analysis says no GPU is required")
    if (pf.test_markers or pf.test_dirs) and not tests.get("command"):
        found = ", ".join(pf.test_markers + pf.test_dirs)
        out.append(f"scan found a test setup ({found}) but the analysis recorded no test command")
    if (pf.hf_ids or pf.weight_urls) and not weights.get("source"):
        out.append("scan found weight references but the analysis recorded no weights source")
    licensing = analysis.get("licensing") or {}
    lic = (licensing.get("code") or analysis.get("license") or "").strip().lower()
    if pf.licence.name and lic and pf.licence.name.lower() not in lic and lic not in pf.licence.name.lower():
        out.append(f"analysis says code licence '{lic}' but {pf.licence.file or 'the LICENSE file'} "
                   f"reads as '{pf.licence.name}' - the file wins")
    if pf.weights_licence.blocking and licensing.get("commercial_use") == "yes":
        out.append(f"analysis says commercial use is permitted, but {pf.weights_licence.file or 'the weights terms'} "
                   f"reads as restricted ({pf.weights_licence.reason})")
    status = (analysis.get("metadata") or {}).get("status")
    if getattr(pf, "archived_hint", "") and status in ("active", "maintained"):
        out.append(f"analysis calls the project active, but the README says '{pf.archived_hint}'")
    days = getattr(pf, "days_since_commit", None)
    if days is not None and days > 730 and status in ("active", "maintained"):
        out.append(f"analysis calls the project active, but the last commit was {days} days ago")
    if getattr(pf, "notebooks", None) and not any(
            (s.get("kind") == "notebook") for s in analysis.get("scripts") or []):
        out.append(f"scan found {len(pf.notebooks)} notebook(s) that the analysis does not mention")
    return out


def review(analysis: dict, pf: Any, workspace: Path, *, hit_step_limit: bool) -> list[str]:
    """Risk lines for the approval gate, most severe first. Empty means clean."""
    risks: list[str] = []

    if hit_step_limit:
        risks.append("ANALYSIS INCOMPLETE: the analyst hit its step limit and was forced to answer early. "
                     "Fields below may be guesses rather than findings - read the report before approving.")

    gaps = completeness_gaps(analysis)
    if gaps:
        risks.append("ANALYSIS INCOMPLETE: " + "; ".join(gaps))

    missing = missing_paths(analysis, workspace)
    if missing:
        shown = ", ".join(f"{p} ({field})" for field, p in missing[:8])
        more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
        risks.append(f"HALLUCINATED PATH: {len(missing)} cited file(s) do not exist in the repo: {shown}{more}. "
                     "Treat the analysis as unreliable.")

    broken = getattr(pf, "broken_scripts", None) or []
    if broken:
        shown = "; ".join(f"{b['path']} ({b['error']})" for b in broken[:5])
        more = f" and {len(broken) - 5} more" if len(broken) > 5 else ""
        risks.append(f"BROKEN SCRIPT: {len(broken)} Python file(s) do not parse: {shown}{more}. "
                     "They cannot run on this interpreter as written.")

    for c in contradictions(analysis, pf):
        risks.append(f"CONTRADICTS SCAN: {c}")

    return risks
