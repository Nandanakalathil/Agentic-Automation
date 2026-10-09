"""Graph routing, approval interrupts and retry loops with all nodes stubbed (no AWS, no LLM)."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from pipeline import graph as graph_mod
from pipeline.config import load_config


class StubNodes:
    """Mimics Nodes' state contract; behaviour is driven by the `script` dict."""

    def __init__(self, cfg, script):
        self.s = script
        self.calls = []
        from pipeline.nodes import Nodes
        self._real = Nodes(cfg)

    def _log(self, name):
        self.calls.append(name)

    def setup_run(self, st):
        self._log("setup_run")
        return {"status": "running", "errors": [], "artifacts": {}, "test_fix_attempts": 0, "adapter_fix_attempts": 0}

    def analyze_repo(self, st):
        self._log("analyze_repo")
        return {"analysis": {"summary": "x"}}

    def plan_ec2(self, st):
        self._log("plan_ec2")
        return {"ec2_plan": {"instance_type": "g5.xlarge", "feedback_seen": st.get("ec2_feedback")}, "ec2_feedback": None}

    def approve_ec2(self, st):
        return self._real.approve_ec2(st)

    def provision_ec2(self, st):
        self._log("provision_ec2")
        return {"ec2_instance_id": "i-123"}

    def run_ec2_tests(self, st):
        self._log("run_ec2_tests")
        n = st["test_fix_attempts"] + 1
        status = self.s["ec2_statuses"][n - 1]
        return {"ec2_test_results": {"status": status}, "test_fix_attempts": n}

    def ec2_test_summary(self, st):
        self._log("ec2_test_summary")
        return {}

    def plan_sagemaker(self, st):
        self._log("plan_sagemaker")
        return {"sagemaker_plan": {"instance_type": "ml.g5.xlarge"}, "sagemaker_feedback": None}

    def approve_sagemaker(self, st):
        return self._real.approve_sagemaker(st)

    def build_adapter(self, st):
        self._log("build_adapter")
        n = st["adapter_fix_attempts"] + 1
        return {"adapter_results": {"verified_contract_passed": self.s["adapter_ok"][n - 1]}, "adapter_fix_attempts": n}

    def deploy_endpoint(self, st):
        self._log("deploy_endpoint")
        return {"endpoint_name": "mdp-x"}

    def configure_scaling(self, st):
        self._log("configure_scaling")
        return {"autoscaling": {}}

    def test_endpoint(self, st):
        self._log("test_endpoint")
        return {"endpoint_test_results": {"passed": self.s.get("endpoint_ok", True)}}

    def finalize(self, st):
        self._log("finalize")
        ok = (st.get("endpoint_test_results") or {}).get("passed")
        return {"status": st.get("status") if st.get("status") != "running" else ("completed" if ok else "failed")}


@pytest.fixture
def make_app(monkeypatch):
    cfg = load_config()

    def _make(script):
        stub = StubNodes(cfg, script)
        monkeypatch.setattr(graph_mod, "Nodes", lambda c: stub)
        return graph_mod.build_graph(cfg, InMemorySaver()), stub
    return _make


T = {"configurable": {"thread_id": "t1"}, "recursion_limit": 100}
INIT = {"run_id": "r", "repo_url": "https://x/y.git", "run_dir": "/tmp/r"}


def test_happy_path_with_two_approvals_and_one_test_retry(make_app):
    app, stub = make_app({"ec2_statuses": ["failed", "passed"], "adapter_ok": [False, True]})
    out = app.invoke(INIT, T)
    assert out["__interrupt__"][0].value["stage"] == "ec2"
    out = app.invoke(Command(resume={"decision": "approve"}), T)
    assert out["__interrupt__"][0].value["stage"] == "sagemaker"
    assert out["__interrupt__"][0].value["ec2_test_status"] == "passed"
    out = app.invoke(Command(resume={"decision": "approve", "instance_type": "ml.g5.2xlarge"}), T)
    assert out["status"] == "completed"
    assert out["sagemaker_plan"]["instance_type"] == "ml.g5.2xlarge"
    assert stub.calls.count("run_ec2_tests") == 2 and stub.calls.count("build_adapter") == 2
    assert stub.calls[-1] == "finalize"


def test_revise_loops_back_to_planner_with_feedback(make_app):
    app, stub = make_app({"ec2_statuses": ["passed"], "adapter_ok": [True]})
    app.invoke(INIT, T)
    out = app.invoke(Command(resume={"decision": "revise", "feedback": "use a cheaper GPU"}), T)
    assert out["__interrupt__"][0].value["stage"] == "ec2"
    assert stub.calls.count("plan_ec2") == 2
    assert app.get_state(T).values["ec2_plan"]["feedback_seen"] == "use a cheaper GPU"


def test_reject_goes_to_finalize_without_provisioning(make_app):
    app, stub = make_app({"ec2_statuses": ["passed"], "adapter_ok": [True]})
    app.invoke(INIT, T)
    out = app.invoke(Command(resume={"decision": "reject"}), T)
    assert out["status"] == "rejected"
    assert "provision_ec2" not in stub.calls


def test_disallowed_override_reprompts_and_run_stays_recoverable(make_app):
    """Regression: raising after interrupt() used to wedge the run permanently."""
    app, stub = make_app({"ec2_statuses": ["passed"], "adapter_ok": [True]})
    app.invoke(INIT, T)
    out = app.invoke(Command(resume={"decision": "approve", "instance_type": "p4d.24xlarge"}), T)
    payload = out["__interrupt__"][0].value
    assert payload["stage"] == "ec2" and "not in allowed" in payload["error"]
    assert "provision_ec2" not in stub.calls
    out = app.invoke(Command(resume={"decision": "approve", "instance_type": "g5.2xlarge"}), T)
    assert out["__interrupt__"][0].value["stage"] == "sagemaker"
    assert app.get_state(T).values["ec2_plan"]["instance_type"] == "g5.2xlarge"


def test_revise_without_feedback_still_replans(make_app):
    """Regression: empty feedback used to route to finalize and fail the run."""
    app, stub = make_app({"ec2_statuses": ["passed"], "adapter_ok": [True]})
    app.invoke(INIT, T)
    out = app.invoke(Command(resume={"decision": "revise", "feedback": ""}), T)
    assert out["__interrupt__"][0].value["stage"] == "ec2"
    assert stub.calls.count("plan_ec2") == 2 and "finalize" not in stub.calls


def test_garbage_answer_reprompts(make_app):
    app, _ = make_app({"ec2_statuses": ["passed"], "adapter_ok": [True]})
    app.invoke(INIT, T)
    out = app.invoke(Command(resume="yes please"), T)
    assert "invalid answer" in out["__interrupt__"][0].value["error"]


def test_adapter_retry_budget_exhausted_ends_failed(make_app):
    app, stub = make_app({"ec2_statuses": ["passed"], "adapter_ok": [False, False, False]})
    app.invoke(INIT, T)
    app.invoke(Command(resume={"decision": "approve"}), T)
    out = app.invoke(Command(resume={"decision": "approve"}), T)
    assert out["status"] == "failed"
    assert stub.calls.count("build_adapter") == 3 and "deploy_endpoint" not in stub.calls
