"""Single-agent harness: the cheap path for developing an agent.

These tests never call an LLM or AWS - they check that inputs are resolved and
that a misuse fails with a message telling you what to run instead, before any
credits are spent.
"""
import json

import pytest

from pipeline import dev


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    from pipeline.config import load_config
    c = load_config()
    monkeypatch.setattr(type(c), "runs_dir", property(lambda self: tmp_path), raising=False)
    return c


def make_run(tmp_path, name="r1", **files):
    d = tmp_path / name
    (d / "repo").mkdir(parents=True)
    for fn, obj in files.items():
        (d / fn).write_text(json.dumps(obj))
    return d


def test_every_agent_maps_to_a_real_node_method():
    from pipeline.nodes import Nodes
    for name, (method, _needs, _ec2) in dev.AGENTS.items():
        assert hasattr(Nodes, method), f"{name} -> {method} does not exist"


def test_free_agents_are_the_ones_needing_no_instance():
    assert set(dev.FREE) == {"repo_analyst", "ec2_planner", "sagemaker_planner"}


def test_unknown_agent_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown agent"):
        dev.run_agent_standalone(cfg, "nope", repo_url="https://x/y")


def test_missing_run_directory_is_reported(cfg):
    with pytest.raises(ValueError, match="no run directory"):
        dev.run_agent_standalone(cfg, "ec2_planner", from_run="absent")


def test_no_inputs_at_all_explains_both_options(cfg):
    with pytest.raises(ValueError, match="--repo .* or --from-run"):
        dev.run_agent_standalone(cfg, "ec2_planner")


def test_missing_upstream_state_names_the_agent_to_run_first(cfg, tmp_path):
    make_run(tmp_path)
    with pytest.raises(ValueError) as e:
        dev.run_agent_standalone(cfg, "ec2_planner", from_run="r1")
    assert "needs analysis" in str(e.value)
    assert "agent repo_analyst --from-run r1" in str(e.value)


def test_ec2_agent_without_an_instance_refuses_before_spending(cfg, tmp_path):
    make_run(tmp_path, **{"analysis.json": {"summary": "x"},
                          "sagemaker_plan.json": {"instance_type": "ml.g5.xlarge"}})
    with pytest.raises(ValueError, match="live EC2 instance"):
        dev.run_agent_standalone(cfg, "adapter_builder", from_run="r1")


def test_state_is_rebuilt_from_per_stage_json(cfg, tmp_path):
    """A run stopped at a gate has no state.json - only the per-stage files."""
    d = make_run(tmp_path, **{"analysis.json": {"summary": "found it"},
                              "ec2_plan.json": {"instance_type": "g5.xlarge"}})
    state = dev._load_state(d)
    assert state["analysis"]["summary"] == "found it"
    assert state["ec2_plan"]["instance_type"] == "g5.xlarge"


def test_state_json_wins_over_per_stage_files(cfg, tmp_path):
    d = make_run(tmp_path, **{"state.json": {"analysis": {"summary": "newer"}},
                              "analysis.json": {"summary": "older"}})
    assert dev._load_state(d)["analysis"]["summary"] == "newer"


def test_instance_id_is_threaded_through(cfg, tmp_path, monkeypatch):
    make_run(tmp_path, **{"analysis.json": {"summary": "x"},
                          "ec2_plan.json": {"instance_type": "g5.xlarge"}})
    seen = {}
    monkeypatch.setattr("pipeline.nodes.Nodes.run_ec2_tests",
                        lambda self, state: seen.update(state) or {"ok": True})
    dev.run_agent_standalone(cfg, "test_fixer", from_run="r1", instance_id="i-0abc")
    assert seen["ec2_instance_id"] == "i-0abc"
    assert seen["test_fix_attempts"] == 0


def test_update_is_persisted_for_the_next_agent(cfg, tmp_path, monkeypatch):
    d = make_run(tmp_path, **{"analysis.json": {"summary": "x"}})
    monkeypatch.setattr("pipeline.nodes.Nodes.plan_ec2",
                        lambda self, state: {"ec2_plan": {"instance_type": "g5.xlarge"}})
    dev.run_agent_standalone(cfg, "ec2_planner", from_run="r1")
    saved = json.loads((d / "state.json").read_text())
    assert saved["ec2_plan"]["instance_type"] == "g5.xlarge"
