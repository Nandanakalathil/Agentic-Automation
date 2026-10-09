"""The free-tier ledger. Wrong numbers here are worse than none, so the
arithmetic, the persistence and the warning thresholds are all pinned."""
import json

import pytest

from pipeline import credits


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    from pipeline.config import load_config
    c = load_config()
    monkeypatch.setattr(type(c), "runs_dir", property(lambda self: tmp_path), raising=False)
    return c


def test_starts_empty(cfg):
    led = credits.load(cfg)
    assert led.used == 0 and led.remaining == led.budget and not led.exhausted


def test_each_call_costs_one_credit(cfg):
    led = credits.load(cfg)
    for _ in range(5):
        led.record(stage="repo_analysis", run_id="r1", model="m", input_tokens=100, output_tokens=10)
    assert led.used == 5
    assert led.total.input_tokens == 500
    assert led.by_stage["repo_analysis"]["calls"] == 5
    assert led.by_run["r1"]["calls"] == 5


def test_survives_a_restart(cfg, tmp_path):
    led = credits.load(cfg)
    led.record(stage="s", run_id="r", model="m")
    led.save()
    assert credits.load(cfg).used == 1


def test_a_corrupt_ledger_does_not_crash_the_run(cfg, tmp_path):
    (tmp_path / credits.LEDGER_NAME).write_text("{not json")
    assert credits.load(cfg).used == 0


def test_remaining_never_goes_negative(cfg):
    cfg["llm"]["free_tier_credits"] = 3
    led = credits.load(cfg)
    for _ in range(10):
        led.record(stage="s", run_id="r", model="m")
    assert led.remaining == 0 and led.exhausted


def test_summary_reports_runs_left(cfg):
    cfg["llm"]["free_tier_credits"] = 300
    assert "~2 full runs" in credits.load(cfg).summary()


def test_config_is_the_source_of_truth_for_the_budget(cfg):
    """Editing free_tier_credits must take effect on an existing ledger."""
    credits.load(cfg).save()
    cfg["llm"]["free_tier_credits"] = 42
    assert credits.load(cfg).budget == 42


# ------------------------------------------------------------- the warnings
def test_no_warning_with_plenty_left(cfg):
    assert credits.preflight_check(cfg, 35, "Agent 1") == ""


def test_warns_when_a_run_would_not_fit(cfg):
    cfg["llm"]["free_tier_credits"] = 1000
    led = credits.load(cfg)
    for _ in range(950):
        led.record(stage="s", run_id="r", model="m")
    led.save()
    msg = credits.preflight_check(cfg, 138, "A full pipeline run")
    assert "only 50 remain" in msg
    assert "EC2 instance billing" in msg      # the consequence that costs real money


def test_warns_when_it_would_eat_over_half(cfg):
    cfg["llm"]["free_tier_credits"] = 100
    assert "over half" in credits.preflight_check(cfg, 60, "Agent 1")


def test_exhausted_is_stated_plainly(cfg):
    cfg["llm"]["free_tier_credits"] = 10
    led = credits.load(cfg)
    for _ in range(10):
        led.record(stage="s", run_id="r", model="m")
    led.save()
    assert "exhausted" in credits.preflight_check(cfg, 35, "Agent 1").lower()


def test_small_agent_warning_omits_the_ec2_consequence(cfg):
    """A 35-call analyst run cannot strand an instance; do not imply it can."""
    cfg["llm"]["free_tier_credits"] = 20
    assert "EC2" not in credits.preflight_check(cfg, 35, "Agent 1")


# -------------------------------------------------------------- the tracker
def test_tracker_counts_a_call_and_its_tokens(cfg):
    class Resp:
        llm_output = {"token_usage": {"prompt_tokens": 1200, "completion_tokens": 80}}
    credits.CreditTracker(cfg, stage="repo_analysis", run_id="r1", model="m").on_llm_end(Resp())
    led = credits.load(cfg)
    assert led.used == 1 and led.total.input_tokens == 1200


def test_tracker_handles_a_response_without_usage(cfg):
    class Resp:
        llm_output = None
    credits.CreditTracker(cfg, stage="s", run_id="r", model="m").on_llm_end(Resp())
    assert credits.load(cfg).used == 1


def test_accounting_never_breaks_a_run(cfg):
    """A ledger failure must not take down a pipeline that is otherwise working."""
    credits.CreditTracker(cfg, stage="s", run_id="r", model="m").on_llm_end(object())


def test_planned_calls_cover_every_dev_agent():
    from pipeline.dev import AGENTS
    assert set(AGENTS) <= set(credits.PLANNED_CALLS)
