"""Deterministic markdown renderers: same layout for every run/model."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

_JSON_FENCE_TAIL = re.compile(r"```json\s*\{.*\}\s*```\s*$", re.S)


def narrative(agent_text: str) -> str:
    """Agent's markdown report without the trailing JSON block."""
    return _JSON_FENCE_TAIL.sub("", agent_text).strip()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _header(title: str, state: dict) -> str:
    return (f"# {title}\n\n| | |\n|---|---|\n| Repository | {state.get('repo_url')} |\n"
            f"| Run ID | `{state.get('run_id')}` |\n| Generated | {_now()} |\n\n")


def _json_appendix(obj) -> str:
    return f"\n\n<details><summary>Machine-readable data</summary>\n\n```json\n{json.dumps(obj, indent=2)}\n```\n</details>\n"


def write(run_dir: str, name: str, content: str) -> str:
    p = Path(run_dir) / name
    p.write_text(content)
    return str(p)


_SEVERITY_ICON = {"blocker": "BLOCKER", "warning": "WARNING", "info": "INFO"}
_SEVERITY_ORDER = {"blocker": 0, "warning": 1, "info": 2}

# Mermaid node shapes by step kind. The model supplies ids, labels and edges;
# the syntax is generated here, because LLMs emit malformed diagram source often
# enough that a hand-built diagram is the difference between a page that renders
# and a page that shows an error box.
_FLOW_SHAPE = {
    "input": '{id}(["{label}"]):::input',
    "output": '{id}(["{label}"]):::output',
    "process": '{id}["{label}"]:::process',
    "model": '{id}[["{label}"]]:::model',
    "decision": '{id}{{"{label}"}}:::decision',
    "external": '{id}[("{label}")]:::external',
}
_FLOW_STYLES = (
    "    classDef input fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;\n"
    "    classDef output fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;\n"
    "    classDef process fill:#fafafa,stroke:#616161,color:#212121;\n"
    "    classDef model fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c;\n"
    "    classDef decision fill:#fff8e1,stroke:#f9a825,color:#e65100;\n"
    "    classDef external fill:#fbe9e7,stroke:#d84315,color:#bf360c;\n"
)


def _mm_id(raw: str) -> str:
    """Mermaid node ids must be bare words."""
    s = re.sub(r"\W", "_", str(raw)).strip("_")
    return s or "step"


def _mm_label(raw: str) -> str:
    return str(raw).replace('"', "'").replace("\n", " ").strip()[:80] or "?"


def flow_diagram(flow: list[dict]) -> str:
    """Render the analyst's flow steps as a Mermaid flowchart. Empty when there are none."""
    if not flow:
        return ""
    known = {_mm_id(s.get("id") or s.get("label", "")) for s in flow}
    lines, edges = [], []
    for step in flow:
        sid = _mm_id(step.get("id") or step.get("label", ""))
        kind = step.get("kind") if step.get("kind") in _FLOW_SHAPE else "process"
        lines.append("    " + _FLOW_SHAPE[kind].format(id=sid, label=_mm_label(step.get("label", sid))))
        for nxt in step.get("next") or []:
            target = _mm_id(nxt)
            if target in known:  # a dangling edge breaks the render; drop it
                edges.append(f"    {sid} --> {target}")
    return "```mermaid\nflowchart TD\n" + "\n".join(lines + edges) + "\n" + _FLOW_STYLES + "```"


def structure_block(tree: list[str]) -> str:
    """The repo layout, straight from the filesystem scan - never model-generated."""
    return "```\n" + "\n".join(tree) + "\n```" if tree else "_(not scanned)_"


_MACHINE_PREFIXES = ("LICENCE", "ANALYSIS INCOMPLETE", "HALLUCINATED PATH", "CONTRADICTS SCAN",
                     "BROKEN SCRIPT")


def _quickstart_block(q: dict) -> str:
    """The first thing a new reader needs, so it sits above everything else."""
    if not q or not (q.get("install") or q.get("run")):
        return ""
    out = "## Quickstart\n\n"
    if q.get("install"):
        out += "**1. Install**\n```bash\n" + "\n".join(q["install"]) + "\n```\n\n"
    if q.get("get_weights"):
        out += "**2. Get the weights**\n```bash\n" + "\n".join(q["get_weights"]) + "\n```\n\n"
    if q.get("run"):
        out += f"**3. Run it**\n```bash\n{q['run']}\n```\n\n"
        if q.get("example_input"):
            out += f"Example input: `{q['example_input']}`\n\n"
        if q.get("expected_output"):
            out += f"Expected result: {q['expected_output']}\n\n"
    if q.get("time_estimate"):
        out += f"Roughly {q['time_estimate']}.\n\n"
    if q.get("caveats"):
        out += "**Watch out for:**\n" + "\n".join(f"- {c}" for c in q["caveats"]) + "\n\n"
    return out


def _alerts_banner(alerts: list[dict], risks: list[str]) -> str:
    """Everything the reader must see before anything else.

    Kept in two groups on purpose. The analyst's alerts are judgement and can be
    wrong; the checked findings were produced by reading the files directly, so a
    reader needs to know which is which rather than weighing them equally.
    """
    reported = []
    for a in sorted(alerts, key=lambda x: _SEVERITY_ORDER.get(x.get("severity", "warning"), 1)):
        icon = _SEVERITY_ICON.get(a.get("severity", "warning"), "WARNING")
        detail = f" — {a['detail']}" if a.get("detail") else ""
        reported.append(f"> **{icon}: {a.get('title', '')}**{detail}")
    checked = [f"> **{r}**" for r in risks if any(r.startswith(p) for p in _MACHINE_PREFIXES)]

    if not reported and not checked:
        return ("> **No blocking issues found.** Review the licensing and compute sections "
                "before deploying.\n")
    out = "## Read this first\n"
    if checked:
        out += "\n**Verified by direct file inspection — these are facts, not judgement:**\n\n"
        out += "\n>\n".join(checked) + "\n"
    if reported:
        out += "\n**Reported by the analyst:**\n\n" + "\n>\n".join(reported) + "\n"
    return out


def _files_table(scripts: list[dict], key_files: list[dict]) -> str:
    """Runnable entry points and the modules behind them, in one table.

    They answer different questions - what do I type, versus where does the
    logic live - so `Kind` keeps that distinction rather than losing it to the
    merge. Scripts come first because that is the order someone reads in, and a
    path appearing in both (a dispatcher is both runnable and the logic) is
    listed once with its invocation.
    """
    seen, out = set(), []
    for s in scripts:
        path = s.get("path", "")
        seen.add(path)
        out.append(f"| `{path}` | {s.get('kind') or 'script'} | {s.get('purpose', '')} "
                   f"| {_code(s.get('invocation'))} |")
    for f in key_files:
        path = f.get("path", "")
        if path in seen:
            continue
        out.append(f"| `{path}` | module | {f.get('role', '')} | - |")
    return "\n".join(out) or "| _none recorded_ | | | |"


def _io_table(items: list[dict], what: str, *, show_required: bool = True) -> str:
    """`required` is meaningless for an output, so the column is dropped there."""
    if not items:
        return f"_No {what} recorded._\n"
    req = (lambda i: f"{'yes' if i.get('required', True) else 'no'} | ") if show_required else (lambda i: "")
    head = "| Name | Format | " + ("Required | " if show_required else "") + "Description | Constraints | Example |\n"
    sep = "|---|---|" + ("---|" if show_required else "") + "---|---|---|\n"
    rows = "\n".join(
        f"| `{i.get('name', '')}` | {i.get('format') or '-'} | {req(i)}"
        f"{i.get('description', '')} | {i.get('constraints') or '-'} | {_code(i.get('example'))} |"
        for i in items)
    return head + sep + rows + "\n"


def _code(v) -> str:
    s = str(v or "").replace("|", "\\|").replace("\n", " ")
    return f"`{s[:80]}`" if s else "-"


def repo_analysis(state: dict, analysis: dict, agent_text: str) -> str:
    a = analysis
    m, lic, hw = a.get("metadata") or {}, a.get("licensing") or {}, a.get("hardware") or {}
    t, w, inf = a.get("tests") or {}, a.get("weights") or {}, a.get("inference") or {}
    pf = a.get("preflight") or {}

    links = a.get("metadata", {}).get("links") or []
    by_kind: dict[str, list[str]] = {}
    for l in links:
        by_kind.setdefault(l.get("kind", "other"), []).append(l.get("url", ""))
    for kind, urls in (pf.get("links") or {}).items():       # scanned links, as a fallback
        by_kind.setdefault(kind, []).extend(u for u in urls if u not in by_kind.get(kind, []))

    def link_cell(*kinds: str) -> str:
        urls = [u for k in kinds for u in by_kind.get(k, [])][:4]
        return ", ".join(f"<{u}>" for u in urls) or "-"

    meta_rows = [
        ("Model", m.get("model_name") or pf.get("repo_name") or "-"),
        ("Status", _status(m, pf)),
        ("Owner", m.get("owner") or pf.get("owner") or "-"),
        ("Category", m.get("category") or "-"),
        ("Purpose", m.get("purpose") or a.get("task") or "-"),
        ("Details", m.get("details") or a.get("summary") or "-"),
        ("Model size", m.get("model_size") or (f"{w.get('approx_size_gb')} GB" if w.get("approx_size_gb")
                                               else _bundled_size(pf))),
        ("Peak memory", _mem(hw)),
        ("Compute", _compute(hw)),
        # Required inputs only: an optional knob is not part of "what this model takes".
        ("Input", ", ".join(i.get("format") or i.get("name", "") for i in a.get("inputs") or []
                            if i.get("required", True)) or inf.get("input_format") or "-"),
        ("Output", ", ".join(o.get("format") or o.get("name", "") for o in a.get("outputs") or []) or
                   inf.get("output_format") or "-"),
        ("Publication", m.get("publication") or "-"),
        ("Paper / preprint", link_cell("arxiv", "biorxiv", "doi", "paper", "nature")),
        ("GitHub", link_cell("github")),
        ("HuggingFace", link_cell("huggingface") or "-"),
        ("Colab / demo", link_cell("colab")),
        ("Code licence", lic.get("code") or a.get("license") or "-"),
        ("Weights licence", lic.get("weights") or "-"),
        ("Commercial use", lic.get("commercial_use") or "unclear"),
        ("Last commit", pf.get("last_commit") or "-"),
    ]
    meta = "\n".join(f"| **{k}** | {str(v).replace(chr(10), ' ')} |" for k, v in meta_rows)

    caps = "\n".join(
        f"| {c.get('task', '')} | `{c.get('entrypoint', '')}` | {c.get('inputs', '')} | {c.get('outputs', '')} "
        f"| {c.get('supported', 'yes')} | {c.get('notes', '')} |"
        for c in a.get("capabilities") or []) or "| _none recorded_ | | | | | |"

    files_table = _files_table(a.get("scripts") or [], a.get("key_files") or [])

    assets = "\n".join(
        f"| `{s.get('path', '')}` | {s.get('kind', '')} | {s.get('size_mb') if s.get('size_mb') is not None else '-'} "
        f"| {'in repo' if s.get('bundled', True) else 'download'} | {s.get('description', '')} |"
        for s in a.get("assets") or []) or "| _none recorded_ | | | | |"

    # `or []`, not a get() default: LLM-repaired JSON sets missing lists to null,
    # which a default never catches.
    fns = "\n".join(f"| `{f['file']}` | `{f['name']}` | `{f.get('signature', '')}` | {f.get('purpose', '')} |"
                    for f in a.get("functions") or []) or "| _none_ | | | |"

    flow_md = flow_diagram(a.get("flow") or [])
    return (
        _header("Repository Analysis", state)
        + _alerts_banner(a.get("alerts") or [], a.get("risks") or [])
        + "\n"
        + _quickstart_block(a.get("quickstart") or {})
        + f"## Metadata\n| | |\n|---|---|\n{meta}\n\n"
        f"## What it can do\n| Task | Entrypoint | Inputs | Outputs | Supported | Notes |\n"
        f"|---|---|---|---|---|---|\n{caps}\n\n"
        + (f"## Process flow\n{flow_md}\n\n" if flow_md else "")
        + f"## Repository structure\n{structure_block(pf.get('tree') or [])}\n\n"
        f"## Inputs\n{_io_table(a.get('inputs') or [], 'inputs')}\n"
        f"## Outputs\n{_io_table(a.get('outputs') or [], 'outputs', show_required=False)}\n"
        f"## Inference entrypoint\n| | |\n|---|---|\n| File | `{inf.get('file', '')}` |\n"
        f"| Function/class | `{inf.get('function_or_class', '')}` |\n| How to call | {inf.get('how_to_call', '')} |\n"
        f"| Input format | {inf.get('input_format', '')} |\n| Output format | {inf.get('output_format', '')} |\n\n"
        f"## Files\n| Path | Kind | Purpose | How to run |\n|---|---|---|---|\n{files_table}\n\n"
        f"## Assets\n| Path | Kind | Size (MB) | Availability | Description |\n|---|---|---|---|---|\n{assets}\n\n"
        f"## Compute requirements\n| | |\n|---|---|\n"
        f"| GPU required | {hw.get('gpu_required')} |\n| Runs on CPU | {hw.get('cpu_only_supported')} |\n"
        f"| Min GPU memory | {hw.get('min_gpu_memory_gb') or '-'} GB |\n| Peak memory | {_mem(hw)} |\n"
        f"| Min RAM | {hw.get('min_ram_gb') or '-'} GB |\n| Disk | {hw.get('disk_gb') or '-'} GB |\n"
        f"| Typical runtime | {hw.get('approx_runtime') or '-'} |\n| Notes | {hw.get('notes', '')} |\n\n"
        f"## Weights\n- Source: {w.get('source', '')}\n- Location/ID: `{w.get('location_or_id', '')}`\n"
        f"- Size: {w.get('approx_size_gb')} GB\n- Download: {w.get('download_instructions', '')}\n\n"
        f"## Licensing\n| | |\n|---|---|\n| Code | {lic.get('code') or a.get('license') or '-'} "
        f"({lic.get('code_file') or '-'}) |\n| Weights | {lic.get('weights') or '-'} "
        f"({lic.get('weights_source') or '-'}) |\n| Commercial use | {lic.get('commercial_use') or 'unclear'} |\n"
        f"| Attribution required | {lic.get('attribution_required', False)} |\n| Notes | {lic.get('notes', '')} |\n\n"
        f"## Tests\n- Framework: {t.get('framework')}\n- Paths: {', '.join(t.get('paths', []))}\n"
        f"- Command: `{t.get('command')}`\n- Approx. count: {t.get('approx_count')}\n"
        f"- Needs GPU: {t.get('needs_gpu')} / needs weights: {t.get('needs_weights')}\n\n"
        f"## Functions and classes\n| File | Name | Signature | Purpose |\n|---|---|---|---|\n{fns}\n\n"
        f"## Dependencies\n- Files: {', '.join(a.get('dependency_files', []))}\n"
        f"- Notable: {', '.join(a.get('notable_dependencies', []))}\n"
        f"- Python: {a.get('python_version') or '-'}; frameworks: {', '.join(a.get('frameworks', [])) or '-'}\n\n"
        + _extra_risks(a.get("risks") or [], a.get("alerts") or [])
    )


def _extra_risks(risks: list[str], alerts: list[dict]) -> str:
    """The free-form risk list, only when there is no structured alert list.

    Models fill `alerts` and `risks` with the same findings in different words,
    so rendering both repeats every item - near-identical text that a literal
    comparison does not catch. `alerts` is the richer of the two (severity,
    title, detail), so when it exists the free-form list is dropped entirely.
    """
    if alerts:
        return ""
    rest = [r for r in risks if not any(r.startswith(p) for p in _MACHINE_PREFIXES)]
    return "## Other notes\n" + "\n".join(f"- {r}" for r in rest) + "\n" if rest else ""


def _mem(hw: dict) -> str:
    peak = hw.get("peak_memory_gb")
    if peak is None:
        return "-"
    basis = f" (source: {hw['peak_memory_basis']})" if hw.get("peak_memory_basis") else ""
    return f"{peak} GB{basis}"


def _status(m: dict, pf: dict) -> str:
    """The analyst's verdict, annotated with the commit date it has to agree with."""
    status = m.get("status") or "unknown"
    days = pf.get("days_since_commit")
    if days is None:
        return status
    years = days / 365.25
    age = f"{days} days" if days < 365 else f"{years:.1f} years"
    return f"{status} (last commit {age} ago)"


def _bundled_size(pf: dict) -> str:
    """Floor for model size when nothing else stated one: weight files actually in the repo."""
    mb = pf.get("bundled_weights_mb") or 0
    return f"{round(mb / 1000, 2)} GB bundled in repo" if mb else "-"


def _compute(hw: dict) -> str:
    if hw.get("gpu_required"):
        vram = f", >= {hw['min_gpu_memory_gb']} GB VRAM" if hw.get("min_gpu_memory_gb") else ""
        return f"GPU required{vram}"
    return "CPU is sufficient" if hw.get("cpu_only_supported") else "unknown"


def ec2_plan(state: dict, plan: dict, hourly: float | None, agent_text: str) -> str:
    cost = f"~${hourly:.2f}/h, ~${hourly * plan['expected_duration_min'] / 60:.2f} for this run" if hourly else "unknown"
    cmds = lambda xs: "\n".join(xs) or "# none"  # noqa: E731
    return (
        _header("EC2 Deployment Plan", state)
        + f"| | |\n|---|---|\n| Instance type | `{plan['instance_type']}` |\n| Estimated cost | {cost} |\n"
        f"| Root volume | {plan['root_volume_gb']} GB |\n| Expected duration | {plan['expected_duration_min']} min |\n\n"
        f"**Rationale:** {plan['rationale']}\n\n"
        f"## Setup\n```bash\n{cmds(plan['setup_commands'])}\n```\n\n"
        f"## Weights\n```bash\n{cmds(plan['weights_commands'])}\n```\n\n"
        f"## Tests\n```bash\n{plan['test_command']}\n```\n\n"
        f"## Smoke inference\n```bash\n{plan['smoke_inference_command']}\n```\n\n"
        f"## Risks\n" + "\n".join(f"- {r}" for r in plan.get("risks", [])) + "\n\n"
        f"## Planner notes\n{narrative(agent_text)}\n" + _json_appendix(plan)
    )


def ec2_test_summary(state: dict, report: dict, agent_text: str) -> str:
    t = report["totals"]
    badge = {"passed": "PASSED", "partial": "PARTIAL", "failed": "FAILED", "no_tests": "NO TESTS"}[report["status"]]
    fails = "\n".join(f"| `{f['test']}` | {f['reason']} | {'yes' if f['resolved'] else 'no'} |"
                      for f in report.get("failures", [])) or "| - | - | - |"
    fixes = "\n".join(f"| `{f['file_or_command']}` | {f['change']} | {f['reason']} |"
                      for f in report.get("fixes_applied", [])) or "| - | - | - |"
    env = "\n".join(f"| {k} | `{v}` |" for k, v in report.get("environment", {}).items())
    return (
        _header("EC2 Test Summary", state)
        + f"## Result: **{badge}**\n\n| Total | Passed | Failed | Skipped | Errors |\n|---|---|---|---|---|\n"
        f"| {t['total']} | {t['passed']} | {t['failed']} | {t['skipped']} | {t['errors']} |\n\n"
        f"- Instance: `{state.get('ec2_plan', {}).get('instance_type')}` (`{state.get('ec2_instance_id')}`)\n"
        f"- Command: `{report.get('test_command')}`\n- Attempts: {state.get('test_fix_attempts', 1)}\n\n"
        f"## Smoke inference\n{report.get('smoke_inference') or '-'}\n\n"
        f"## Failures\n| Test | Reason | Resolved |\n|---|---|---|\n{fails}\n\n"
        f"## Fixes applied (reproducibility log)\n| File / command | Change | Reason |\n|---|---|---|\n{fixes}\n\n"
        f"## Environment\n| Component | Version |\n|---|---|\n{env}\n\n"
        f"## Notes\n{report.get('notes', '')}\n\n## Agent narrative\n{narrative(agent_text)}\n" + _json_appendix(report)
    )


def sagemaker_plan(state: dict, plan: dict, hourly: float | None, agent_text: str) -> str:
    a = plan["autoscaling"]
    lo = f"${hourly * 730 * a['min_capacity']:,.0f}" if hourly else "?"
    hi = f"${hourly * 730 * a['max_capacity']:,.0f}" if hourly else "?"
    return (
        _header("SageMaker Deployment Plan", state)
        + f"| | |\n|---|---|\n| Instance type | `{plan['instance_type']}` |\n"
        f"| Hourly per instance | {f'~${hourly:.2f}' if hourly else 'unknown'} |\n"
        f"| Monthly (always-on) | {lo} at min capacity - {hi} at max capacity |\n"
        f"| Initial instances | {plan['initial_instance_count']} |\n| Base image | `{plan['base_image']}` |\n"
        f"| Weights strategy | {plan['weights_strategy']} (`{plan.get('model_artifact_dir') or '-'}`) |\n"
        f"| Max concurrency / instance | {plan['max_concurrency']} |\n| Expected latency | {plan.get('expected_latency_ms')} ms |\n"
        f"| Autoscaling | min {a['min_capacity']}, max {a['max_capacity']}, target {a['target_invocations_per_instance']} "
        f"invocations/instance/min |\n\n**Rationale:** {plan['rationale']}\n\n"
        f"## Risks\n" + "\n".join(f"- {r}" for r in plan.get("risks", [])) + "\n\n"
        f"## Planner notes\n{narrative(agent_text)}\n" + _json_appendix(plan)
    )


def endpoint_test_summary(state: dict, results: dict) -> str:
    c, load = results["contract"], results.get("load", {})
    lat = load.get("latency_ms", {})
    failed = [x for x in c["cases"] if not x["passed"]]
    rows = "\n".join(f"| {x['name']} | {x['kind']} | {'PASS' if x['passed'] else 'FAIL'} | "
                     f"{x.get('latency_ms') and round(x['latency_ms'])} | {x['detail']} |" for x in c["cases"])
    sc = state.get("autoscaling", {})
    verdict = "PASSED" if not failed and load.get("error_rate_pct", 0) == 0 else "FAILED"
    return (
        _header("SageMaker Endpoint Test Summary", state)
        + f"## Result: **{verdict}**\n\n| | |\n|---|---|\n| Endpoint | `{state.get('endpoint_name')}` |\n"
        f"| Instance type | `{state.get('sagemaker_plan', {}).get('instance_type')}` |\n"
        f"| Contract tests | {c['passed']}/{c['total']} passed |\n"
        f"| Load test | {load.get('requests')} requests @ concurrency {load.get('concurrency')} |\n"
        f"| Throughput | {load.get('throughput_rps')} req/s |\n"
        f"| Latency p50 / p95 / p99 | {lat.get('p50')} / {lat.get('p95')} / {lat.get('p99')} ms |\n"
        f"| Error rate | {load.get('error_rate_pct')}% |\n\n"
        f"## Autoscaling\n- Min/max instances: {sc.get('min')} / {sc.get('max')}\n"
        f"- Target: {sc.get('target_invocations_per_instance_per_min')} invocations/instance/min\n"
        f"- Alarms: {', '.join(sc.get('alarms', []))}\n\n"
        f"## Contract cases\n| Case | Kind | Result | ms | Detail |\n|---|---|---|---|---|\n{rows}\n"
        + _json_appendix(results)
    )


def integration_guide(state: dict, schema: dict, sample: dict | None, region: str) -> str:
    ep = state.get("endpoint_name")
    sample_json = json.dumps(sample or {"inputs": ["..."]}, indent=2)
    return (
        _header("Endpoint Integration Guide", state)
        + f"## Endpoint\n| | |\n|---|---|\n| Name | `{ep}` |\n| Region | `{region}` |\n"
        f"| Protocol | SageMaker InvokeEndpoint, `application/json` |\n| Contract | `{schema.get('$id', '')}` |\n\n"
        "## IAM permission for the calling backend\n```json\n"
        + json.dumps({"Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
                      "Resource": f"arn:aws:sagemaker:{region}:<account-id>:endpoint/{ep}"}, indent=2)
        + "\n```\n\n## Python (boto3)\n```python\nimport boto3, json\n\n"
        f"smr = boto3.client('sagemaker-runtime', region_name='{region}')\n"
        f"resp = smr.invoke_endpoint(EndpointName='{ep}', ContentType='application/json',\n"
        f"                           Body=json.dumps({json.dumps(sample or {})}))\n"
        "result = json.loads(resp['Body'].read())\n```\n\n"
        f"## Example request\n```json\n{sample_json}\n```\n\n"
        "## Errors\nErrors return HTTP 4xx/5xx with the schema's `error` object. Through boto3 they surface as "
        "`ModelError` with `OriginalStatusCode` and `OriginalMessage` (the JSON error body).\n\n"
        f"## Schema\n```json\n{json.dumps(schema, indent=2)}\n```\n"
    )


def final_report(state: dict) -> str:
    arts = "\n".join(f"- [{k}]({Path(v).name})" for k, v in sorted((state.get("artifacts") or {}).items()))
    errs = "\n".join(f"- {e}" for e in state.get("errors") or []) or "- none"
    return (
        _header("Pipeline Run Report", state)
        + f"**Status:** {state.get('status')}  \n**Endpoint:** `{state.get('endpoint_name') or '-'}`  \n"
        f"**EC2 instance:** `{state.get('ec2_instance_id') or '-'}` (terminated after build unless noted)\n\n"
        f"## Artifacts\n{arts}\n\n## Errors / warnings\n{errs}\n"
    )
