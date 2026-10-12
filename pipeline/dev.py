"""Run one agent on its own, against saved inputs.

Developing an agent through full pipeline runs is the expensive way to do it: a
run costs ~138 API calls and, past the first gate, real AWS time. Each node is
already a plain method on `Nodes` taking a state dict, so a node can be driven
directly once its inputs exist.

The inputs come from a previous run directory, so iterating on the EC2 planner
re-uses the analysis the analyst produced rather than regenerating it. Only the
node under test spends credits.

    python -m pipeline agent repo_analyst      --repo https://github.com/o/r
    python -m pipeline agent ec2_planner       --from-run <run_id>
    python -m pipeline agent sagemaker_planner --from-run <run_id>
    python -m pipeline agent test_fixer        --from-run <run_id> --instance i-0abc
    python -m pipeline agent adapter_builder   --from-run <run_id> --instance i-0abc

`test_fixer` and `adapter_builder` need a live EC2 instance. Pass `--instance`
to re-use one you already have: provisioning a fresh box per iteration is what
makes developing those two agents expensive, not the agent itself.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .nodes import STATE_DIR, Nodes
from .state import PipelineState

# agent name -> (node method, state keys it needs beyond run_dir/workspace, needs an instance)
AGENTS: dict[str, tuple[str, list[str], bool]] = {
    "repo_analyst":      ("analyze_repo",      [],                            False),
    "ec2_planner":       ("plan_ec2",          ["analysis"],                  False),
    "test_fixer":        ("run_ec2_tests",     ["analysis", "ec2_plan"],      True),
    "sagemaker_planner": ("plan_sagemaker",    ["analysis"],                  False),
    "adapter_builder":   ("build_adapter",     ["analysis", "sagemaker_plan"], True),
}

FREE = [n for n, (_, _, needs_ec2) in AGENTS.items() if not needs_ec2]

# Position in the pipeline, 0-indexed, used for the output path. Kept explicit
# rather than derived from AGENTS order so renaming or reordering an agent does
# not silently move everyone's output directory.
AGENT_NUMBER: dict[str, int] = {
    "repo_analyst": 0,
    "ec2_planner": 1,
    "test_fixer": 2,
    "sagemaker_planner": 3,
    "adapter_builder": 4,
}


def repo_slug(repo_url: str) -> str:
    """'https://github.com/biomed-AI/GraphEC.git' -> 'GraphEC'.

    Case is preserved: the directory should read like the project it holds.
    """
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "repo"


def output_dir(cfg: Config, agent: str, repo_url: str, run: str) -> Path:
    """outputs/<repo>/agent<n>/<run> - one repo's history in one place."""
    return cfg.runs_dir / repo_slug(repo_url) / f"agent{AGENT_NUMBER[agent]}" / run


def next_run_name(parent: Path) -> str:
    """'run1', 'run2', ... one past the highest that exists.

    Deliberately not the first free number: filling a gap would put a new run
    before older ones in a listing, and the whole point of the sequence is that
    it reads in the order the runs happened.
    """
    highest = 0
    if parent.exists():
        for p in parent.glob("run*"):
            m = re.fullmatch(r"run(\d+)", p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"run{highest + 1}"


def _load_state(run_dir: Path) -> PipelineState:
    """Rebuild state from a previous run's artefacts.

    state.json is written only by finalize, so a run stopped at a gate has none.
    The per-stage JSON files are written as each node completes, which is
    exactly the case this tool is for.
    """
    state: dict[str, Any] = {}
    sj = run_dir / STATE_DIR / "state.json"
    if sj.exists():
        state.update(json.loads(sj.read_text()))
    for key, name in (("analysis", "analysis.json"),
                      ("ec2_plan", "ec2_plan.json"),
                      ("ec2_test_results", "ec2_test_results.json"),
                      ("sagemaker_plan", "sagemaker_plan.json")):
        p = run_dir / STATE_DIR / name
        if p.exists() and key not in state:
            state[key] = json.loads(p.read_text())
    return state  # type: ignore[return-value]


def _new_run_dir(cfg: Config, repo_url: str, agent: str) -> tuple[str, Path]:
    """Allocate outputs/<repo>/agent<n>/run<k> for a fresh run.

    Named by position rather than timestamp: the point of the layout is that
    one repo's runs sit together in order and can be diffed against each other.
    """
    parent = cfg.runs_dir / repo_slug(repo_url) / f"agent{AGENT_NUMBER[agent]}"
    run = next_run_name(parent)
    run_dir = parent / run
    return str(run_dir.relative_to(cfg.runs_dir)), run_dir


def run_agent_standalone(cfg: Config, agent: str, *, repo_url: str | None = None,
                         from_run: str | None = None, instance_id: str | None = None,
                         repo_ref: str | None = None, run_id: str | None = None) -> dict:
    """Execute one node and return its state update. Raises ValueError on bad inputs."""
    if agent not in AGENTS:
        raise ValueError(f"unknown agent '{agent}'. Choose from: {', '.join(AGENTS)}")
    method, needs, needs_ec2 = AGENTS[agent]
    nodes = Nodes(cfg)

    if from_run:
        run_dir = cfg.runs_dir / from_run
        if not run_dir.is_dir():
            raise ValueError(f"no run directory at {run_dir}")
        state = _load_state(run_dir)
        state.setdefault("run_id", from_run)
        state.setdefault("run_dir", str(run_dir))
        state.setdefault("workspace", str(run_dir / "repo"))
        state.setdefault("repo_url", repo_url or state.get("repo_url", ""))
    else:
        if not repo_url:
            raise ValueError("pass --repo <url> (to start fresh) or --from-run <run_id>")
        if run_id:
            if any(part in ("..", "") for part in Path(run_id).parts):
                raise ValueError(f"invalid --run-id {run_id!r}")
            # A bare name ("run3") is placed under this repo's agent directory;
            # a path ("GraphEC/agent0/run3") is taken as given, so an existing
            # run can still be targeted exactly.
            run_dir = (output_dir(cfg, agent, repo_url, run_id) if len(Path(run_id).parts) == 1
                       else cfg.runs_dir / run_id)
            run_id = str(run_dir.relative_to(cfg.runs_dir))
        else:
            run_id, run_dir = _new_run_dir(cfg, repo_url, agent)
        state = {"run_id": run_id, "repo_url": repo_url, "repo_ref": repo_ref, "run_dir": str(run_dir)}
        print(f"[dev] cloning into {run_dir}")
        state.update(nodes.setup_run(state))  # type: ignore[arg-type]

    missing = [k for k in needs if not state.get(k)]
    if missing:
        raise ValueError(
            f"'{agent}' needs {', '.join(missing)} in state, which this run does not have.\n"
            f"Run the earlier agent first, e.g.:\n"
            f"  python -m pipeline agent {_producer(missing[0])} --from-run {state.get('run_id')}")

    if needs_ec2:
        if not instance_id and not state.get("ec2_instance_id"):
            raise ValueError(
                f"'{agent}' runs commands on a live EC2 instance.\n"
                f"Pass --instance i-xxxx to re-use one you already have, or run the full\n"
                f"pipeline to provision one. Re-using a single instance across iterations is\n"
                f"what keeps developing this agent cheap.")
        if instance_id:
            state["ec2_instance_id"] = instance_id

    # Defaults the retry-loop nodes read but a standalone invocation has not set.
    state.setdefault("test_fix_attempts", 0)
    state.setdefault("adapter_fix_attempts", 0)
    state.setdefault("artifacts", {})
    state.setdefault("errors", [])

    Path(state["run_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"[dev] running '{agent}' ({method}) in {state['run_dir']}")
    update = getattr(nodes, method)(state)

    merged = {**state, **update}
    sj = Path(state["run_dir"]) / STATE_DIR / "state.json"
    sj.parent.mkdir(parents=True, exist_ok=True)
    sj.write_text(json.dumps(merged, indent=2, default=str))
    return update


def _producer(state_key: str) -> str:
    """Which agent produces a given state key, for the error message."""
    for name, (_m, _needs, _e) in AGENTS.items():
        if (state_key == "analysis" and name == "repo_analyst") \
                or (state_key == "ec2_plan" and name == "ec2_planner") \
                or (state_key == "sagemaker_plan" and name == "sagemaker_planner"):
            return name
    return "repo_analyst"
