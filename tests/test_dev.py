"""Single-agent harness: the cheap path for developing an agent.

These tests never call an LLM or AWS - they check that inputs are resolved and
that a misuse fails with a message telling you what to run instead, before any
credits are spent.
"""
import json

import pytest

from pipeline import dev
from pipeline.nodes import STATE_DIR


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    from pipeline.config import load_config
    c = load_config()
    monkeypatch.setattr(type(c), "runs_dir", property(lambda self: tmp_path), raising=False)
    return c


def make_run(tmp_path, name="r1", **files):
    """A previous run: the report at the top, machine state in .state/."""
    d = tmp_path / name
    (d / "repo").mkdir(parents=True)
    (d / STATE_DIR).mkdir(parents=True)
    for fn, obj in files.items():
        (d / STATE_DIR / fn).write_text(json.dumps(obj))
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
    saved = json.loads((d / STATE_DIR / "state.json").read_text())
    assert saved["ec2_plan"]["instance_type"] == "g5.xlarge"


def test_the_run_directory_shows_only_the_report(cfg, tmp_path, monkeypatch):
    """One run, one deliverable. Working files exist but stay out of the way."""
    d = make_run(tmp_path, **{"analysis.json": {"summary": "x"}})
    monkeypatch.setattr("pipeline.nodes.Nodes.plan_ec2", lambda self, state: {"ec2_plan": {}})
    dev.run_agent_standalone(cfg, "ec2_planner", from_run="r1")
    visible = sorted(p.name for p in d.iterdir() if not p.name.startswith("."))
    assert visible == ["repo"]          # no state.json, no analysis.json at the top
    assert (d / STATE_DIR / "state.json").exists()


# --- outputs/<repo>/agent<n>/run<k> ------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://github.com/biomed-AI/GraphEC", "GraphEC"),
    ("https://github.com/biomed-AI/GraphEC.git", "GraphEC"),
    ("https://github.com/biomed-AI/GraphEC/", "GraphEC"),
    ("git@github.com:facebookresearch/esm.git", "esm"),
    ("https://huggingface.co/org/my model", "my-model"),
])
def test_repo_slug_keeps_the_project_name_readable(url, expected):
    assert dev.repo_slug(url) == expected


class _Cfg(dict):
    """Minimal stand-in: output_dir only ever touches runs_dir."""
    def __init__(self, root):
        super().__init__()
        self.runs_dir = root


def test_output_dir_shape(tmp_path):
    cfg = _Cfg(tmp_path)
    url = "https://github.com/biomed-AI/GraphEC"
    assert dev.output_dir(cfg, "repo_analyst", url, "run1") == tmp_path / "GraphEC" / "agent0" / "run1"
    assert dev.output_dir(cfg, "ec2_planner", url, "run1") == tmp_path / "GraphEC" / "agent1" / "run1"


def test_agent_numbers_are_unique_and_cover_every_agent():
    assert set(dev.AGENT_NUMBER) == set(dev.AGENTS)
    assert sorted(dev.AGENT_NUMBER.values()) == list(range(len(dev.AGENTS)))


def test_next_run_name_skips_what_exists(tmp_path):
    assert dev.next_run_name(tmp_path / "nothing-here") == "run1"
    (tmp_path / "run1").mkdir()
    (tmp_path / "run2").mkdir()
    assert dev.next_run_name(tmp_path) == "run3"
    (tmp_path / "run4").mkdir()          # a gap is left alone, not filled
    assert dev.next_run_name(tmp_path) == "run5"
    (tmp_path / "runX").mkdir()          # a non-numbered sibling is ignored
    assert dev.next_run_name(tmp_path) == "run5"
