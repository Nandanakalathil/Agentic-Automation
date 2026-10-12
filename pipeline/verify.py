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


# "fold0-4.ckpt" means five real files; "{0..4}" and "<ID>.txt" are the same
# habit. These are imprecise, not invented, and calling them hallucinations
# puts "treat the analysis as unreliable" on top of an otherwise sound report -
# which teaches the reader to skip the banner that matters.
_SHORTHAND = re.compile(r"\d+\s*-\s*\d+|\{.*\}|<[^>]+>|\bN\b")


def _is_shorthand(path: str, workspace: Path) -> bool:
    """True when a path is a range/placeholder whose directory really exists."""
    if not _SHORTHAND.search(Path(path).name):
        return False
    parent = workspace / Path(path).parent
    return parent.is_dir() and any(parent.iterdir())


def missing_paths(analysis: dict, workspace: Path) -> list[tuple[str, str]]:
    """Cited paths that do not exist in the cloned repo, deduplicated.

    Range shorthand is excluded here and reported separately by `shorthand_paths`.
    """
    seen: set[str] = set()
    missing: list[tuple[str, str]] = []
    for field, p in _cited_paths(analysis):
        if p in seen:
            continue
        seen.add(p)
        if not (workspace / p).exists() and not _is_shorthand(p, workspace):
            missing.append((field, p))
    return missing


def shorthand_paths(analysis: dict, workspace: Path) -> list[tuple[str, str]]:
    """Cited paths that stand for several real files. Downstream agents copy
    these strings into commands, so `cp fold0-4.ckpt` would fail."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for field, p in _cited_paths(analysis):
        if p in seen:
            continue
        seen.add(p)
        if not (workspace / p).exists() and _is_shorthand(p, workspace):
            out.append((field, p))
    return out


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
    if hw.get("gpu_required") and not hw.get("peak_memory_gb"):
        gaps.append("no peak memory estimate - the planner cannot size the instance; "
                    "an estimate with its basis was asked for, not a blank")
    if not analysis.get("functions"):
        gaps.append("no call path recorded - the adapter builder cannot follow the inference chain")
    return gaps


# Flags that configure a run rather than feed the model. If one of these is
# listed as a model input, it ends up in the serving schema, and an API client
# is asked to supply a GPU index.
_EXECUTION_FLAGS = {
    "gpu", "gpus", "device", "cuda", "seed", "batch_size", "batchsize", "num_workers",
    "output_dir", "outdir", "out", "verbose", "debug", "cpu", "threads", "n_jobs",
}


def _io_problems(analysis: dict) -> list[str]:
    out: list[str] = []
    for spec in analysis.get("inputs") or []:
        name = str(spec.get("name", "")).strip().lower().lstrip("-").replace("-", "_")
        if name in _EXECUTION_FLAGS:
            out.append(f"'{name}' is listed as a model input, but it is an execution flag - "
                       "it would end up in the serving schema")
    outputs = [str(o.get("name", "")).strip().lower() for o in analysis.get("outputs") or []]
    formats = {"text", "txt", "json", "csv", "file", "string", "tensor", "binary"}
    if len(outputs) > 1 and all(o in formats for o in outputs):
        out.append("outputs are named by format rather than by artefact - "
                   "a reader cannot tell the results apart")
    return out


# Licence names vary in punctuation, version and the word "licence": Apache-2.0,
# "Apache License 2.0" and "Apache License, Version 2.0" are one licence. Compare
# the family word only, or every Apache repo reports a false contradiction.
_LICENCE_NOISE = re.compile(r"\b(licen[sc]e|version|v?\d+(\.\d+)*|the|of|public)\b|[^a-z0-9 ]")


def _licence_key(name: str) -> set[str]:
    return set(_LICENCE_NOISE.sub(" ", (name or "").lower()).split())


def _same_licence(a: str, b: str) -> bool:
    ka, kb = _licence_key(a), _licence_key(b)
    return bool(ka and kb and (ka & kb))


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
    if pf.licence.name and lic and not _same_licence(lic, pf.licence.name):
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
    # The scan read these out of the README; an analysis that records none has
    # not looked at the one place this repo states its environment.
    if getattr(pf, "readme_pins", None) and not analysis.get("version_pins"):
        src = pf.readme_pins[0].get("source", "the README")
        out.append(f"scan read {len(pf.readme_pins)} version pin(s) from {src} "
                   "that the analysis did not record")
    bins = getattr(pf, "bundled_binaries", None) or []
    if bins and not (analysis.get("hardware") or {}).get("platform_constraint"):
        out.append(f"scan found {len(bins)} prebuilt binary/-ies ({bins[0]['path']} is "
                   f"{bins[0]['format']}) but no platform constraint was recorded - "
                   "the deployment is locked to that OS/arch")
    out += _io_problems(analysis)
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

    shorthand = shorthand_paths(analysis, workspace)
    if shorthand:
        shown = ", ".join(p for _f, p in shorthand[:6])
        risks.append(f"IMPRECISE PATH: {len(shorthand)} path(s) are written as a range or placeholder "
                     f"rather than a real filename: {shown}. The files exist; expand the name before "
                     "putting it in a command.")

    broken = getattr(pf, "broken_scripts", None) or []
    if broken:
        shown = "; ".join(f"{b['path']} ({b['error']})" for b in broken[:5])
        more = f" and {len(broken) - 5} more" if len(broken) > 5 else ""
        risks.append(f"BROKEN SCRIPT: {len(broken)} Python file(s) do not parse: {shown}{more}. "
                     "They cannot run on this interpreter as written.")

    for c in contradictions(analysis, pf):
        risks.append(f"CONTRADICTS SCAN: {c}")

    return risks
