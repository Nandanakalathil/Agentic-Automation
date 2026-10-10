"""CLI.

  python -m pipeline run <git-url> [--ref BRANCH_OR_SHA] [--no-input]
  python -m pipeline resume <run-id> [--no-input]
  python -m pipeline approve <run-id> --decision approve|revise|reject [--instance-type T] [--feedback TEXT]
  python -m pipeline status <run-id>
  python -m pipeline list
  python -m pipeline cleanup [--run-id ID] [--endpoints] [--yes]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import load_config
from .graph import build_graph


def _app(cfg):
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.runs_dir / "checkpoints.sqlite", check_same_thread=False)
    return build_graph(cfg, SqliteSaver(conn))


def _thread(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 100}


def _prompt_approval(payload: dict) -> dict:
    stage = payload["stage"].upper()
    print("\n" + "=" * 72)
    print(f" APPROVAL REQUIRED: {stage} deployment")
    print("=" * 72)
    if payload.get("error"):
        print(f" ! previous answer not accepted: {payload['error']}")
    print(f" Plan report   : {payload.get('report')}")
    print(f" Instance type : {payload['instance_type']}  (~${payload.get('hourly_usd') or '?'}/hour)")
    if payload.get("ec2_test_status"):
        print(f" EC2 tests     : {payload['ec2_test_status']}")
    print(f" Allowed types : {', '.join(payload['allowed_instance_types'])}")
    for f in payload.get("findings") or []:
        print(f" ! {f}")
    print(" Read the report, then choose:")
    print("   [a] approve   [i] approve with a different instance type   [r] revise with feedback   [x] reject")
    while True:
        c = input(" > ").strip().lower()
        if c == "a":
            return {"decision": "approve"}
        if c == "i":
            t = input(" instance type: ").strip()
            if t in payload["allowed_instance_types"]:
                return {"decision": "approve", "instance_type": t}
            print(" not in the allowed list")
        elif c == "r":
            fb = input(" feedback for the planner: ").strip()
            if fb:
                return {"decision": "revise", "feedback": fb}
            print(" feedback is required for a revision")
        elif c == "x":
            return {"decision": "reject"}


def _report_credits(run_id: str = "") -> None:
    """Always shown after work finishes: the free tier is small enough to track per run."""
    try:
        from . import credits
        from .config import load_config
        led = credits.load(load_config())
        if run_id:
            line = led.per_run_summary(run_id)
            if line:
                print(line)
        print(led.summary())
    except Exception:
        pass


def _drive(app, run_id: str, inp, interactive: bool) -> int:
    cfg_thread = _thread(run_id)
    while True:
        try:
            result = app.invoke(inp, cfg_thread)
        except KeyboardInterrupt:
            print(f"\nInterrupted. Resume with: python -m pipeline resume {run_id}")
            return 130
        except Exception as e:
            print(f"\n[pipeline] ERROR in run {run_id}: {type(e).__name__}: {e}")
            print(f"  Fix the cause, then resume:  python -m pipeline resume {run_id}")
            print(f"  Or tear down resources:      python -m pipeline cleanup --run-id {run_id}")
            print("  (The EC2 instance also self-terminates after aws.ec2.self_destruct_hours.)")
            return 1
        interrupts = result.get("__interrupt__") or []
        if not interrupts:
            print(f"\nRun {run_id}: {result.get('status')}. Reports in runs/{run_id}/")
            _report_credits(run_id)
            return 0 if result.get("status") == "completed" else 2
        payload = interrupts[0].value
        if not interactive:
            print(f"\nRun {run_id} is waiting for {payload['stage']} approval. Report: {payload.get('report')}")
            if payload.get("error"):
                print(f"  ! previous answer not accepted: {payload['error']}")
            for f in payload.get("findings") or []:
                print(f"  ! {f}")
            if payload.get("ec2_instance_id"):
                print(f"  NOTE: EC2 instance {payload['ec2_instance_id']} keeps running (and billing) while this "
                      "waits. It self-terminates after aws.ec2.self_destruct_hours.")
            print(f"  python -m pipeline approve {run_id} --decision approve|revise|reject "
                  "[--instance-type T] [--feedback '...']")
            return 0
        inp = Command(resume=_prompt_approval(payload))


def cmd_run(cfg, a) -> int:
    from . import credits
    warn = credits.preflight_check(cfg, credits.PLANNED_CALLS["full_run"], "A full pipeline run")
    if warn:
        print(f"\n  {warn}\n")
        if not a.no_input and input("  Continue anyway? [y/N] ").strip().lower() != "y":
            return 1
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", a.repo_url.rstrip("/").split("/")[-1].removesuffix(".git")).strip("-").lower()
    run_id = a.run_id or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{slug[:30]}"
    run_dir = cfg.runs_dir / run_id
    print(f"[pipeline] run {run_id} -> {run_dir}")
    init = {"run_id": run_id, "repo_url": a.repo_url, "repo_ref": a.ref, "run_dir": str(run_dir)}
    return _drive(_app(cfg), run_id, init, not a.no_input)


def cmd_resume(cfg, a) -> int:
    app = _app(cfg)
    snap = app.get_state(_thread(a.run_id))
    if not snap.values:
        print(f"no checkpoint for run {a.run_id}")
        return 1
    pending = [i for t in snap.tasks for i in t.interrupts]
    if pending and not a.no_input:
        return _drive(app, a.run_id, Command(resume=_prompt_approval(pending[0].value)), True)
    if pending:
        print(f"waiting for approval; use: python -m pipeline approve {a.run_id} ...")
        return 0
    return _drive(app, a.run_id, None, not a.no_input)


def cmd_approve(cfg, a) -> int:
    from .aws import pricing
    from .nodes import Nodes

    if a.decision == "revise" and not a.feedback:
        print("--feedback is required with --decision revise")
        return 1
    app = _app(cfg)
    pending = [i.value for t in app.get_state(_thread(a.run_id)).tasks for i in t.interrupts]
    if not pending:
        print(f"run {a.run_id} is not waiting for an approval")
        return 1
    if a.instance_type:
        stage = pending[0]["stage"]
        section = cfg["aws"][stage]
        price_fn = pricing.ec2_hourly if stage == "ec2" else pricing.sagemaker_hourly
        problem = Nodes._check_instance(a.instance_type, section, price_fn)
        if problem:
            print(f"instance type rejected: {problem}")
            return 1
    answer = {"decision": a.decision}
    if a.instance_type:
        answer["instance_type"] = a.instance_type
    if a.feedback:
        answer["feedback"] = a.feedback
    return _drive(app, a.run_id, Command(resume=answer), not a.no_input)


def cmd_status(cfg, a) -> int:
    snap = _app(cfg).get_state(_thread(a.run_id))
    if not snap.values:
        print(f"no checkpoint for run {a.run_id}")
        return 1
    v = snap.values
    pending = [i.value for t in snap.tasks for i in t.interrupts]
    print(json.dumps({
        "run_id": a.run_id, "status": v.get("status"), "next": list(snap.next),
        "waiting_for_approval": pending[0]["stage"] if pending else None,
        "ec2_instance_id": v.get("ec2_instance_id"), "endpoint_name": v.get("endpoint_name"),
        "ec2_tests": (v.get("ec2_test_results") or {}).get("status"),
        "endpoint_tests_passed": (v.get("endpoint_test_results") or {}).get("passed"),
        "artifacts": v.get("artifacts"), "errors": v.get("errors"),
    }, indent=2))
    return 0


def cmd_list(cfg, a) -> int:
    if not cfg.runs_dir.exists():
        print("no runs yet")
        return 0
    for d in sorted(p for p in cfg.runs_dir.iterdir() if p.is_dir()):
        st = d / "state.json"
        status = json.loads(st.read_text()).get("status") if st.exists() else "in progress / interrupted"
        print(f"{d.name:55s} {status}")
    return 0


def cmd_credits(cfg, a) -> int:
    from . import credits
    led = credits.load(cfg)
    print(led.summary())
    if led.by_stage:
        print("\n  by agent:")
        for stage, u in sorted(led.by_stage.items(), key=lambda kv: -kv[1]["calls"]):
            print(f"    {stage:20} {u['calls']:4} calls")
    if a.reset:
        led.path.unlink(missing_ok=True)
        print("\n  ledger reset.")
    return 0


def cmd_agent(cfg, a) -> int:
    """Run one agent on its own, against saved inputs (see pipeline/dev.py)."""
    from .dev import AGENTS, FREE, run_agent_standalone

    if a.agent == "list":
        print("agents:")
        for name, (method, needs, needs_ec2) in AGENTS.items():
            cost = "needs a live EC2 instance" if needs_ec2 else "no AWS cost"
            print(f"  {name:19} {method:17} needs: {', '.join(needs) or 'nothing':28} {cost}")
        print(f"\nfree to iterate on: {', '.join(FREE)}")
        return 0
    from . import credits
    warn = credits.preflight_check(cfg, credits.PLANNED_CALLS.get(a.agent, 40), f"Agent '{a.agent}'")
    if warn:
        print(f"\n  {warn}\n")
        if input("  Continue anyway? [y/N] ").strip().lower() != "y":
            return 1
    try:
        update = run_agent_standalone(cfg, a.agent, repo_url=a.repo, from_run=a.from_run,
                                      instance_id=a.instance, repo_ref=a.ref, run_id=a.run_id)
    except ValueError as e:
        print(f"[dev] {e}")
        return 1
    print(f"\n[dev] '{a.agent}' produced keys: {', '.join(sorted(update))}")
    for path in (update.get("artifacts") or {}).values():
        print(f"  report: {path}")
    _report_credits()
    return 0


def cmd_cleanup(cfg, a) -> int:
    from .aws.ec2 import find_tagged_instances
    from .aws.sagemaker import SageMakerDeployer
    from .aws.session import TAG_KEY, boto_session

    insts = [i for i in find_tagged_instances(cfg)
             if not a.run_id or any(t["Key"] == "pipeline-run-id" and t["Value"] == a.run_id for t in i.get("Tags", []))]
    endpoints = []
    if a.endpoints:
        sm = boto_session(cfg).client("sagemaker")
        for ep in sm.list_endpoints(NameContains="mdp-", MaxResults=100)["Endpoints"]:
            tags = {t["Key"]: t["Value"] for t in sm.list_tags(ResourceArn=ep["EndpointArn"])["Tags"]}
            if tags.get(TAG_KEY) == "true" and (not a.run_id or tags.get("pipeline-run-id") == a.run_id):
                endpoints.append((ep["EndpointName"], tags.get("pipeline-run-id")))
    print("EC2 instances:", [(i["InstanceId"], i["InstanceType"], i["State"]["Name"]) for i in insts] or "none")
    if a.endpoints:
        print("Endpoints:", endpoints or "none")
    if not insts and not endpoints:
        return 0
    if not a.yes and input("Delete these? [y/N] ").strip().lower() != "y":
        return 0
    if insts:
        boto_session(cfg).client("ec2").terminate_instances(InstanceIds=[i["InstanceId"] for i in insts])
    for name, rid in endpoints:
        SageMakerDeployer(cfg, rid or "cleanup").delete_endpoint(name)
    print("done")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m pipeline")
    p.add_argument("--config", help="path to pipeline.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("repo_url")
    r.add_argument("--ref")
    r.add_argument("--run-id")
    r.add_argument("--no-input", action="store_true", help="stop at approval gates instead of prompting")
    r = sub.add_parser("resume")
    r.add_argument("run_id")
    r.add_argument("--no-input", action="store_true")
    r = sub.add_parser("approve")
    r.add_argument("run_id")
    r.add_argument("--decision", required=True, choices=["approve", "revise", "reject"])
    r.add_argument("--instance-type")
    r.add_argument("--feedback")
    r.add_argument("--no-input", action="store_true")
    sub.add_parser("status").add_argument("run_id")
    sub.add_parser("list")
    r = sub.add_parser("credits", help="show free-tier credit usage")
    r.add_argument("--reset", action="store_true", help="clear the ledger")
    r = sub.add_parser("agent", help="run a single agent against saved inputs (cheap iteration)")
    r.add_argument("agent", help="agent name, or 'list'")
    r.add_argument("--repo", help="clone this repo and start fresh")
    r.add_argument("--from-run", help="reuse a previous run's analysis/plans")
    r.add_argument("--instance", help="existing EC2 instance id for test_fixer / adapter_builder")
    r.add_argument("--ref", help="git ref to check out when using --repo")
    r.add_argument("--run-id", help="name the output folder, e.g. GraphEC/run2")
    r = sub.add_parser("cleanup")
    r.add_argument("--run-id")
    r.add_argument("--endpoints", action="store_true", help="also delete pipeline-created SageMaker endpoints")
    r.add_argument("--yes", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    return {"run": cmd_run, "resume": cmd_resume, "approve": cmd_approve, "status": cmd_status,
            "list": cmd_list, "cleanup": cmd_cleanup, "agent": cmd_agent,
            "credits": cmd_credits}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
