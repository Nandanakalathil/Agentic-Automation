"""LangGraph node implementations. Each node: (state) -> partial state update."""
from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from . import preflight, prompts, reports, verify
from .aws import pricing
from .aws.ec2 import REMOTE_REPO, Ec2Runner
from .aws.sagemaker import SageMakerDeployer, endpoint_name_for
from .config import Config
from .harness import AgentResult, run_agent, to_structured
from .llm import get_llm
from .models import AdapterResult, Ec2Plan, Ec2TestReport, RepoAnalysis, SageMakerPlan
from .state import PipelineState
from .tools.ec2_tools import make_ec2_tools
from .tools.repo_tools import make_repo_tools

SERVING_DIR = Path(__file__).resolve().parent.parent / "serving"
FIXED_SERVING_FILES = ["serve.py", "envelope.py", "contract_tests.py"]
LOCAL_IMAGE = "mdp-local:latest"


def _print(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _step_printer(agent: str):
    def on_step(step: int, ai) -> None:
        calls = ", ".join(f"{c['name']}({_short(c['args'])})" for c in ai.tool_calls) or "final answer"
        _print(f"  {agent} step {step}: {calls}")
    return on_step


def _short(args: dict) -> str:
    s = json.dumps(args)
    return s if len(s) < 90 else s[:87] + "..."


def _resources(state: PipelineState, **kv) -> None:
    """Persist created cloud resources immediately (survives crashes before the node returns)."""
    p = Path(state["run_dir"]) / "resources.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    data.update(kv)
    p.write_text(json.dumps(data, indent=2))


def _repo_slug(url: str) -> str:
    return re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1]) or "model"


class Nodes:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.agent_cfg = cfg["agent"]

    # ---------------------------------------------------------------- helpers
    def _agent(self, state, *, name, stage, system, task, tools, max_steps=None) -> AgentResult:
        _print(f"agent '{name}' starting (model: {self.cfg['llm']['models'][stage]})")
        return run_agent(
            get_llm(self.cfg, stage, state.get("run_id", "")), name=name, system_prompt=system, task=task, tools=tools,
            max_steps=max_steps or self.agent_cfg["max_steps"],
            max_tool_output_chars=self.agent_cfg["max_tool_output_chars"],
            context_char_budget=self.agent_cfg["context_char_budget"],
            log_dir=Path(state["run_dir"]) / "logs", on_step=_step_printer(name),
        )

    def _structured(self, stage, text, model_cls, run_id=""):
        return to_structured(get_llm(self.cfg, stage, run_id), text, model_cls)

    def _runner(self, state) -> Ec2Runner:
        return Ec2Runner(self.cfg, state["run_id"], state.get("ec2_instance_id"))

    def _artifact(self, state, key: str, name: str, content: str) -> dict:
        path = reports.write(state["run_dir"], name, content)
        _print(f"wrote {path}")
        return {**(state.get("artifacts") or {}), key: path}

    @staticmethod
    def _dump(obj: Any, limit: int = 30000) -> str:
        return json.dumps(obj, indent=1)[:limit]

    # ---------------------------------------------------------------- 0. setup
    def setup_run(self, state: PipelineState) -> dict:
        run_dir = Path(state["run_dir"])
        ws = run_dir / "repo"
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (ws / ".git").exists():
            _print(f"cloning {state['repo_url']}")
            env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
            depth = [] if state.get("repo_ref") else ["--depth", "1"]
            # Timeouts: a clone that stalls on a huge LFS repo or an unreachable host
            # would otherwise hang the run forever with no output.
            clone_timeout = self.cfg.get_path("repo.clone_timeout_s", 1800)
            subprocess.run(["git", "clone", *depth, state["repo_url"], str(ws)],
                           check=True, env=env, timeout=clone_timeout)
            if state.get("repo_ref"):
                subprocess.run(["git", "-C", str(ws), "checkout", state["repo_ref"]],
                               check=True, env=env, timeout=clone_timeout)
        sha = subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        _resources(state, repo_url=state["repo_url"], commit=sha)
        return {"workspace": str(ws), "commit_sha": sha, "status": "running", "errors": [], "artifacts": {},
                "test_fix_attempts": 0, "adapter_fix_attempts": 0}

    # ---------------------------------------------------------------- 1. analysis
    def analyze_repo(self, state: PipelineState) -> dict:
        # Deterministic scan first: free, offline, and it stops the analyst spending
        # steps rediscovering requirements.txt. Also the only reliable licence check.
        pf = preflight.scan(Path(state["workspace"]))
        (Path(state["run_dir"]) / "preflight.json").write_text(json.dumps(pf.to_dict(), indent=2))
        _print(f"preflight: {len(pf.build_files)} build files, licence={pf.licence.name or 'unknown'}"
               + (f", bio stack: {', '.join(pf.bio_deps)}" if pf.is_bio else ""))
        warning = preflight.licence_warning(pf)
        if warning:
            _print(warning)

        tools = make_repo_tools(Path(state["workspace"]), writable=False)
        res = self._agent(state, name="repo_analyst", stage="repo_analysis",
                          system=prompts.REPO_ANALYST + prompts.output_contract(RepoAnalysis),
                          task=(f"Analyse the repository cloned from {state['repo_url']} "
                                f"(commit {state.get('commit_sha')}).\n\n{preflight.to_prompt_block(pf)}"),
                          tools=tools,
                          # The analyst reads far more than it acts, and now produces a full
                          # orientation document rather than a deployment checklist.
                          max_steps=self.agent_cfg["max_steps"] + self.cfg.get_path("agent.analyst_extra_steps", 40))
        analysis = self._structured("repo_analysis", res.final_text, RepoAnalysis).model_dump()
        # The scan beats the model on these two: it read the files rather than inferring.
        if pf.licence.name and not analysis.get("license"):
            analysis["license"] = pf.licence.name
        # Schema-valid is not the same as useful: check the analysis against the
        # filesystem and the scan before the gate, since every later stage costs money.
        gate_findings = [*([warning] if warning else []),
                         *verify.review(analysis, pf, Path(state["workspace"]),
                                        hit_step_limit=res.hit_step_limit)]
        for f in gate_findings:
            _print(f)
        analysis["risks"] = [*gate_findings, *(analysis.get("risks") or [])]
        analysis["verification"] = {"hit_step_limit": res.hit_step_limit, "steps": res.steps,
                                    "findings": gate_findings}
        analysis["preflight"] = pf.to_dict()
        (Path(state["run_dir"]) / "analysis.json").write_text(json.dumps(analysis, indent=2))
        md = reports.repo_analysis(state, analysis, res.final_text)
        return {"analysis": analysis, "artifacts": self._artifact(state, "01_repo_analysis", "01_repo_analysis.md", md)}

    # ---------------------------------------------------------------- 2. EC2 plan + approval
    def plan_ec2(self, state: PipelineState) -> dict:
        ec2 = self.cfg["aws"]["ec2"]
        fb = state.get("ec2_feedback")
        system = prompts.EC2_PLANNER.format(
            allowed=", ".join(ec2["allowed_instance_types"]), max_hourly=ec2["max_hourly_usd"],
            feedback=f"\nReviewer feedback on the previous plan (must address): {fb}" if fb else "",
        ) + prompts.output_contract(Ec2Plan)
        task = f"Repository analysis:\n{self._dump(state['analysis'])}\n\nProduce the EC2 deployment plan."
        tools = make_repo_tools(Path(state["workspace"]), writable=False)
        for attempt in range(2):
            res = self._agent(state, name="ec2_planner", stage="ec2_plan", system=system, task=task, tools=tools,
                              max_steps=20)
            plan = self._structured("ec2_plan", res.final_text, Ec2Plan)
            problem = self._check_instance(plan.instance_type, ec2, pricing.ec2_hourly)
            if not problem:
                break
            task += f"\n\nYour previous plan was rejected by a guardrail: {problem}. Choose again."
        else:
            raise ValueError(f"EC2 planner kept violating guardrails: {problem}")
        plan_d = plan.model_dump()
        md = reports.ec2_plan(state, plan_d, pricing.ec2_hourly(plan.instance_type), res.final_text)
        return {"ec2_plan": plan_d, "ec2_feedback": None,
                "artifacts": self._artifact(state, "02_ec2_deployment_plan", "02_ec2_deployment_plan.md", md)}

    @staticmethod
    def _check_instance(itype: str, section: dict, price_fn) -> str | None:
        if itype not in section["allowed_instance_types"]:
            return f"{itype} is not in allowed_instance_types {section['allowed_instance_types']}"
        price = price_fn(itype)
        if price is not None and price > section["max_hourly_usd"]:
            return f"{itype} costs ~${price}/h, above the ${section['max_hourly_usd']}/h cap"
        return None

    def _approval(self, state: PipelineState, *, stage: str, plan_key: str, report_key: str,
                  section: dict, price_fn) -> dict:
        plan = state[plan_key]
        error = None
        # Never raise after interrupt(): LangGraph replays stored resume values on re-execution,
        # so a raise here would repeat on every resume. Invalid answers re-ask instead (each
        # interrupt() call in the loop consumes the next resume value).
        while True:
            answer = interrupt({
                "stage": stage,
                "instance_type": plan["instance_type"],
                "hourly_usd": price_fn(plan["instance_type"]),
                "report": state["artifacts"].get(report_key),
                # Deterministic findings about the analysis the plan rests on. The gate is
                # the last point where stopping is free, so they must be visible here and
                # not only in a report file nobody opens before approving.
                "findings": ((state.get("analysis") or {}).get("verification") or {}).get("findings") or [],
                "allowed_instance_types": section["allowed_instance_types"],
                "ec2_test_status": (state.get("ec2_test_results") or {}).get("status"),
                "ec2_instance_id": state.get("ec2_instance_id") if stage == "sagemaker" else None,
                "error": error,
            })
            # answer: {"decision": "approve"|"revise"|"reject", "instance_type": str?, "feedback": str?}
            decision = answer.get("decision") if isinstance(answer, dict) else None
            if decision not in ("approve", "revise", "reject"):
                error = f"invalid answer {answer!r}; decision must be approve, revise or reject"
                continue
            override = answer.get("instance_type")
            if decision == "approve" and override and override != plan["instance_type"]:
                error = self._check_instance(override, section, price_fn)
                if error:
                    error = f"instance override rejected: {error}"
                    continue
            break
        update: dict[str, Any] = {
            f"{stage}_approved": decision == "approve",
            f"{stage}_decision": decision,
            f"{stage}_feedback": (answer.get("feedback") or "(reviewer asked for a revised plan without details)")
            if decision == "revise" else None,
        }
        if decision == "approve" and override and override != plan["instance_type"]:
            update[plan_key] = {**plan, "instance_type": override}
        if decision == "reject":
            update["status"] = "rejected"
            update["errors"] = (state.get("errors") or []) + [f"{stage} plan rejected by reviewer"]
        return update

    def approve_ec2(self, state: PipelineState) -> dict:
        return self._approval(state, stage="ec2", plan_key="ec2_plan", report_key="02_ec2_deployment_plan",
                              section=self.cfg["aws"]["ec2"], price_fn=pricing.ec2_hourly)

    # ---------------------------------------------------------------- 3. EC2 provision + tests
    def _ensure_ec2(self, state: PipelineState, *, replay_setup: bool) -> tuple[Ec2Runner, bool]:
        """Return a runner for a live, bootstrapped instance; launch one if needed.

        replay_setup re-runs the approved setup/weights commands, for an instance that
        died (self-destruct TTL, manual termination) after the test stage.
        """
        runner = self._runner(state)
        if runner.is_alive():
            return runner, False
        plan = state["ec2_plan"]
        if state.get("ec2_instance_id"):
            _print(f"EC2 instance {state['ec2_instance_id']} is gone; launching a replacement")
        iid = runner.launch(plan["instance_type"], plan.get("root_volume_gb"))
        _resources(state, ec2_instance_id=iid)
        _print(f"launched {plan['instance_type']} {iid}; waiting for SSM")
        runner.wait_until_ready()
        boot = runner.bootstrap()
        _print(f"bootstrap: {boot['stdout'][-300:].strip()}")
        _print(runner.sync_workspace(Path(state["workspace"])))
        if replay_setup:
            for cmd in plan.get("setup_commands", []) + plan.get("weights_commands", []):
                r = runner.run(cmd, timeout_s=self.cfg.get_path("aws.ec2.command_timeout_s", 3600))
                _print(f"replay [{r['exit_code']}]: {cmd[:80]}")
        return runner, True

    def provision_ec2(self, state: PipelineState) -> dict:
        runner, launched = self._ensure_ec2(state, replay_setup=False)
        if not launched:
            _print(f"reusing EC2 instance {runner.instance_id}")
            runner.bootstrap()
            _print(runner.sync_workspace(Path(state["workspace"])))
        return {"ec2_instance_id": runner.instance_id}

    def run_ec2_tests(self, state: PipelineState) -> dict:
        runner = self._runner(state)
        ws = Path(state["workspace"])
        tools = make_repo_tools(ws, writable=True) + make_ec2_tools(
            runner, ws, self.cfg.get_path("aws.ec2.command_timeout_s", 3600))
        prev = state.get("last_failure")
        system = prompts.EC2_TESTER.format(
            plan=self._dump(state["ec2_plan"]), analysis=self._dump(state["analysis"], 15000),
            previous=f"\nA previous attempt ended with this report - continue from it, do not start over:\n{prev}" if prev else "",
        ) + prompts.output_contract(Ec2TestReport)
        res = self._agent(state, name=f"ec2_tester_{state.get('test_fix_attempts', 0) + 1}", stage="test_fixer",
                          system=system, task="Set up the environment, run the test suite, fix environmental failures, "
                          "run the smoke inference, then report.", tools=tools,
                          max_steps=self.agent_cfg["max_steps"] * 2)
        report = self._structured("test_fixer", res.final_text, Ec2TestReport).model_dump()
        attempts = state.get("test_fix_attempts", 0) + 1
        _print(f"EC2 tests: {report['status']} {report['totals']} (attempt {attempts})")
        return {"ec2_test_results": {**report, "_agent_text": res.final_text}, "test_fix_attempts": attempts,
                "last_failure": json.dumps(report)[:8000] if report["status"] == "failed" else None}

    def ec2_test_summary(self, state: PipelineState) -> dict:
        r = dict(state["ec2_test_results"])
        text = r.pop("_agent_text", "")
        md = reports.ec2_test_summary(state, r, text)
        return {"last_failure": None,
                "artifacts": self._artifact(state, "03_ec2_test_summary", "03_ec2_test_summary.md", md)}

    # ---------------------------------------------------------------- 4. SageMaker plan + approval
    def plan_sagemaker(self, state: PipelineState) -> dict:
        sm = self.cfg["aws"]["sagemaker"]
        fb = state.get("sagemaker_feedback")
        ec2_report = {k: v for k, v in state["ec2_test_results"].items() if k != "_agent_text"}
        system = prompts.SAGEMAKER_PLANNER.format(
            allowed=", ".join(sm["allowed_instance_types"]), max_hourly=sm["max_hourly_usd"],
            min_cap=sm["autoscaling"]["min_capacity"], max_cap=sm["autoscaling"]["max_capacity"],
            schema=json.dumps(self.cfg.load_schema(), indent=1), ec2_report=self._dump(ec2_report),
            analysis=self._dump(state["analysis"], 15000),
            feedback=f"\nReviewer feedback on the previous plan (must address): {fb}" if fb else "",
        ) + prompts.output_contract(SageMakerPlan)
        tools = make_repo_tools(Path(state["workspace"]), writable=False)
        task = "Produce the SageMaker deployment plan."
        for _ in range(2):
            res = self._agent(state, name="sagemaker_planner", stage="sagemaker_plan", system=system, task=task,
                              tools=tools, max_steps=20)
            plan = self._structured("sagemaker_plan", res.final_text, SageMakerPlan)
            problem = self._check_instance(plan.instance_type, sm, pricing.sagemaker_hourly)
            if not problem:
                break
            task += f"\n\nYour previous plan was rejected by a guardrail: {problem}. Choose again."
        else:
            raise ValueError(f"SageMaker planner kept violating guardrails: {problem}")
        # Clamp autoscaling to configured bounds.
        a = plan.autoscaling
        a.max_capacity = min(a.max_capacity, sm["autoscaling"]["max_capacity"])
        a.min_capacity = max(1, min(a.min_capacity, a.max_capacity))
        plan_d = plan.model_dump()
        md = reports.sagemaker_plan(state, plan_d, pricing.sagemaker_hourly(plan.instance_type), res.final_text)
        return {"sagemaker_plan": plan_d, "sagemaker_feedback": None,
                "artifacts": self._artifact(state, "04_sagemaker_deployment_plan", "04_sagemaker_deployment_plan.md", md)}

    def approve_sagemaker(self, state: PipelineState) -> dict:
        return self._approval(state, stage="sagemaker", plan_key="sagemaker_plan",
                              report_key="04_sagemaker_deployment_plan",
                              section=self.cfg["aws"]["sagemaker"], price_fn=pricing.sagemaker_hourly)

    # ---------------------------------------------------------------- 5. adapter + container contract tests
    def _install_serving_template(self, ws: Path) -> None:
        mdp = ws / ".mdp"
        mdp.mkdir(exist_ok=True)
        for f in FIXED_SERVING_FILES:  # always reset the fixed layer
            shutil.copy(SERVING_DIR / f, mdp / f)
        shutil.copy(self.cfg.project_root / self.cfg["schema_path"], mdp / "endpoint_schema.json")
        if not (mdp / "Dockerfile").exists():
            shutil.copy(SERVING_DIR / "Dockerfile.template", mdp / "Dockerfile")
        if not (mdp / "model_adapter.py").exists():
            shutil.copy(SERVING_DIR / "model_adapter_template.py", mdp / "model_adapter.py")
        if not (mdp / "requirements-serving.txt").exists():
            (mdp / "requirements-serving.txt").write_text("")

    @staticmethod
    def _artifact_dir(plan: dict) -> str | None:
        """Absolute EC2 path of the weights dir (relative paths are relative to the repo)."""
        art = plan.get("model_artifact_dir") if plan.get("weights_strategy") == "model_tar_gz" else None
        if not art:
            return None
        return art if art.startswith("/") else posixpath.normpath(posixpath.join(REMOTE_REPO, art))

    def _container_test_script(self, state: PipelineState) -> str:
        plan = state["sagemaker_plan"]
        art = self._artifact_dir(plan)
        # A relative -v source would silently create an empty named volume.
        mount = f"-v {art}:/opt/ml/model:ro" if art else ""
        check_art = f'[ -d {art} ] || {{ echo "MODEL_ARTIFACT_DIR_MISSING: {art}"; exit 4; }}' if art else ""
        env = " ".join(f"-e {k}={json.dumps(str(v))}" for k, v in (plan.get("environment") or {}).items())
        return f"""
{check_art}
python3 -m pip install -q jsonschema >/dev/null 2>&1
docker build -f .mdp/Dockerfile -t {LOCAL_IMAGE} . > /tmp/mdp_build.log 2>&1 || {{ echo BUILD_FAILED; tail -n 80 /tmp/mdp_build.log; exit 2; }}
docker rm -f mdp-test >/dev/null 2>&1 || true
GPU=""; nvidia-smi >/dev/null 2>&1 && GPU="--gpus all"
docker run -d --name mdp-test -p 8080:8080 $GPU {mount} -e MDP_MAX_CONCURRENCY={plan.get('max_concurrency', 1)} {env} {LOCAL_IMAGE} serve >/dev/null
for i in $(seq 1 180); do
  curl -sf localhost:8080/ping >/dev/null && break
  [ "$(docker inspect -f '{{{{.State.Running}}}}' mdp-test)" = "true" ] || {{ echo CONTAINER_EXITED; docker logs --tail 120 mdp-test 2>&1; exit 3; }}
  sleep 5
done
python3 .mdp/contract_tests.py --url http://localhost:8080 --schema .mdp/endpoint_schema.json \
  --samples .mdp/sample_requests.json --wait-ready 0 --load-requests 20 --load-concurrency 2
echo "----- container logs -----"; docker logs --tail 60 mdp-test 2>&1
docker rm -f mdp-test >/dev/null 2>&1 || true
"""

    def build_adapter(self, state: PipelineState) -> dict:
        ws = Path(state["workspace"])
        self._install_serving_template(ws)
        runner, replaced = self._ensure_ec2(state, replay_setup=True)
        last: dict[str, Any] = {}

        def build_and_test_container() -> str:
            """Sync workspace, docker-build the serving image on EC2, run it, and run the schema contract tests."""
            runner.sync_workspace(ws)
            r = runner.run(self._container_test_script(state), timeout_s=3600)
            out = r["stdout"]
            m = re.search(r"MDP_RESULT_JSON=(\{.*\})", out)
            if m:
                result = json.loads(m.group(1))
                last.clear()
                last.update(result)
                c = result["contract"]
                failed = [x for x in c["cases"] if not x["passed"]]
                logs = out.split("----- container logs -----")[-1][-4000:]
                return json.dumps({"contract_passed": c["failed"] == 0, "passed": c["passed"], "total": c["total"],
                                   "failed_cases": failed[:15], "load": result.get("load"),
                                   "container_logs_tail": logs})
            last.clear()
            return json.dumps({"contract_passed": False, "exit_code": r["exit_code"],
                               "output_tail": out[-6000:], "stderr_tail": r["stderr"][-2000:]})

        from langchain_core.tools import StructuredTool
        tools = (make_repo_tools(ws, writable=True) + make_ec2_tools(runner, ws)
                 + [StructuredTool.from_function(build_and_test_container)])
        ec2_report = {k: v for k, v in state["ec2_test_results"].items() if k != "_agent_text"}
        prev = state.get("last_failure")
        if replaced:
            prev = ((prev or "") + "\nNOTE: the EC2 instance was replaced. The approved setup and weights commands "
                    "were replayed, but check that dependencies and weights are present before building.").strip()
        system = prompts.ADAPTER_BUILDER.format(
            plan=self._dump(state["sagemaker_plan"]), ec2_report=self._dump(ec2_report),
            analysis=self._dump(state["analysis"], 12000), schema=json.dumps(self.cfg.load_schema(), indent=1),
            previous=f"\nThe previous attempt failed - fix this:\n{prev}" if prev else "",
        ) + prompts.output_contract(AdapterResult, with_report=False)
        res = self._agent(state, name=f"adapter_builder_{state.get('adapter_fix_attempts', 0) + 1}",
                          stage="adapter_builder", system=system,
                          task="Write the adapter files, then build and contract-test the container until it passes.",
                          tools=tools, max_steps=self.agent_cfg["max_steps"] * 2)
        claimed = self._structured("adapter_builder", res.final_text, AdapterResult)
        if not last:  # the agent never ran a full test (or the last run crashed): verify ourselves
            _print("verifying container contract tests")
            build_and_test_container()
        verified = bool(last) and last["contract"]["failed"] == 0
        if claimed.contract_passed and not verified:
            _print("agent claimed success but verification failed - not trusting the claim")
        attempts = state.get("adapter_fix_attempts", 0) + 1
        _print(f"container contract tests: {'PASSED' if verified else 'FAILED'} (attempt {attempts})")
        failure = None if verified else json.dumps(
            [c for c in (last.get("contract", {}).get("cases", [])) if not c["passed"]][:15] or claimed.notes)[:8000]
        return {"adapter_results": {**claimed.model_dump(), "verified_contract_passed": verified,
                                    "container_test": last},
                "adapter_fix_attempts": attempts, "last_failure": failure,
                "ec2_instance_id": runner.instance_id}

    # ---------------------------------------------------------------- 6. deploy + scale
    def deploy_endpoint(self, state: PipelineState) -> dict:
        plan = state["sagemaker_plan"]
        runner, replaced = self._ensure_ec2(state, replay_setup=True)
        if replaced:  # the tested image lived on the old instance: rebuild it first
            return {"ec2_instance_id": runner.instance_id,
                    "last_failure": "EC2 instance was replaced before deployment; rebuild and re-test the container.",
                    "adapter_results": {**state["adapter_results"], "verified_contract_passed": False}}
        dep = SageMakerDeployer(self.cfg, state["run_id"])
        name = state.get("endpoint_name") or endpoint_name_for(_repo_slug(state["repo_url"]), state["run_id"])
        tag = f"{state['run_id'][-8:]}-a{state.get('adapter_fix_attempts', 1)}"
        image = dep.push_local_image(runner, LOCAL_IMAGE, tag)
        _print(f"pushed {image}")
        model_data = None
        if self._artifact_dir(plan):
            model_data = dep.package_model_artifacts(runner, self._artifact_dir(plan))
            _print(f"model artifacts at {model_data}")
        dep.delete_endpoint(name)  # clean up a failed previous attempt, if any
        _resources(state, endpoint_name=name, image_uri=image)
        _print(f"creating endpoint {name} on {plan['instance_type']} (this takes 5-15 min)")
        env = {"MDP_MAX_CONCURRENCY": plan.get("max_concurrency", 1), **(plan.get("environment") or {})}
        result = dep.deploy(endpoint_name=name, image_uri=image, model_data_url=model_data,
                            instance_type=plan["instance_type"], initial_count=plan["initial_instance_count"], env=env)
        _print(f"endpoint status: {result['status']}")
        update = {"endpoint_name": name, "image_uri": image}
        if result["status"] != "InService":
            update["last_failure"] = f"SageMaker endpoint {result['status']}: {result.get('reason')}\n{result.get('logs', '')[-6000:]}"
            update["adapter_results"] = {**state["adapter_results"], "verified_contract_passed": False}
        return update

    def configure_scaling(self, state: PipelineState) -> dict:
        dep = SageMakerDeployer(self.cfg, state["run_id"])
        sm = self.cfg["aws"]["sagemaker"]
        a = state["sagemaker_plan"]["autoscaling"]
        scaling = dep.configure_autoscaling(
            state["endpoint_name"], min_capacity=a["min_capacity"], max_capacity=a["max_capacity"],
            target_invocations=a.get("target_invocations_per_instance") or sm["autoscaling"]["target_invocations_per_instance"],
            scale_in_cooldown=sm["autoscaling"]["scale_in_cooldown_s"],
            scale_out_cooldown=sm["autoscaling"]["scale_out_cooldown_s"])
        alarms = dep.create_alarms(state["endpoint_name"], p95_latency_ms=sm["alarms"]["p95_latency_ms"],
                                   error_rate_pct=sm["alarms"]["error_rate_pct"],
                                   sns_topic_arn=sm["alarms"].get("sns_topic_arn"))
        _print(f"autoscaling {a['min_capacity']}-{a['max_capacity']}, alarms: {alarms}")
        return {"autoscaling": {**scaling, "alarms": alarms}, "alarms": alarms}

    # ---------------------------------------------------------------- 7. endpoint tests
    def test_endpoint(self, state: PipelineState) -> dict:
        import importlib.util
        spec = importlib.util.spec_from_file_location("contract_tests", SERVING_DIR / "contract_tests.py")
        ct = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ct)

        dep = SageMakerDeployer(self.cfg, state["run_id"])
        ep = state["endpoint_name"]
        send = lambda payload: dep.invoke(ep, payload)  # noqa: E731
        samples = json.loads((Path(state["workspace"]) / ".mdp" / "sample_requests.json").read_text())
        lt = self.cfg["aws"]["sagemaker"]["load_test"]
        results = {"contract": ct.run_contract_tests(send, None, self.cfg.load_schema(), samples)}
        if samples.get("valid"):
            results["load"] = ct.run_load_test(send, samples["valid"][0], concurrency=lt["concurrency"],
                                               total=lt["requests"])
        (Path(state["run_dir"]) / "endpoint_test_results.json").write_text(json.dumps(results, indent=2))
        md = reports.endpoint_test_summary(state, results)
        passed = results["contract"]["failed"] == 0 and results.get("load", {}).get("error_rate_pct", 0) == 0
        _print(f"endpoint tests: {'PASSED' if passed else 'FAILED'} "
               f"({results['contract']['passed']}/{results['contract']['total']} contract)")
        update = {"endpoint_test_results": {**results, "passed": passed},
                  "artifacts": self._artifact(state, "05_sagemaker_test_summary", "05_sagemaker_test_summary.md", md)}
        if not passed:
            failed = [c for c in results["contract"]["cases"] if not c["passed"]][:15]
            update["last_failure"] = ("Tests against the deployed SageMaker endpoint failed:\n" + json.dumps(failed)
                                      + f"\nload: {json.dumps(results.get('load'))}\nlogs:\n{dep.recent_logs(ep)[-4000:]}")
            update["adapter_results"] = {**state["adapter_results"], "verified_contract_passed": False}
        return update

    # ---------------------------------------------------------------- 8. finalize
    def finalize(self, state: PipelineState) -> dict:
        errors = list(state.get("errors") or [])
        status = state.get("status") or "running"
        ok = bool((state.get("endpoint_test_results") or {}).get("passed"))
        if status == "running":
            status = "completed" if ok else "failed"
            if not ok:
                errors.append(f"pipeline stopped: {(state.get('last_failure') or 'see logs')[:500]}")

        if state.get("ec2_instance_id") and self.cfg.get_path("aws.ec2.auto_teardown", True):
            runner = self._runner(state)
            if runner.is_alive():
                runner.terminate()
                _print(f"terminated EC2 instance {runner.instance_id}")

        if status == "failed" and state.get("endpoint_name") and self.cfg.get_path("aws.sagemaker.delete_endpoint_on_failure"):
            SageMakerDeployer(self.cfg, state["run_id"]).delete_endpoint(state["endpoint_name"])
            errors.append(f"endpoint {state['endpoint_name']} deleted after failure (delete_endpoint_on_failure)")

        artifacts = dict(state.get("artifacts") or {})
        if status == "completed":
            samples_p = Path(state["workspace"]) / ".mdp" / "sample_requests.json"
            sample = json.loads(samples_p.read_text())["valid"][0] if samples_p.exists() else None
            md = reports.integration_guide(state, self.cfg.load_schema(), sample, self.cfg.get_path("aws.region"))
            artifacts["06_endpoint_integration"] = reports.write(state["run_dir"], "06_endpoint_integration.md", md)
            # Keep the exact serving files that went into the image next to the reports.
            shutil.copytree(Path(state["workspace"]) / ".mdp", Path(state["run_dir"]) / "serving_bundle",
                            dirs_exist_ok=True)

        final = {**state, "status": status, "errors": errors, "artifacts": artifacts}
        reports.write(state["run_dir"], "00_final_report.md", reports.final_report(final))
        (Path(state["run_dir"]) / "state.json").write_text(json.dumps(final, indent=2, default=str))
        _print(f"run {state['run_id']} finished: {status}")
        return {"status": status, "errors": errors, "artifacts": artifacts}
