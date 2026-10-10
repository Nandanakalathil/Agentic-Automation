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


# ------------------------------------------------------------- the API key
def test_nvidia_api_key_is_read(monkeypatch):
    from pipeline.config import load_config
    from pipeline.llm import get_llm
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    assert get_llm(load_config(), "repo_analysis") is not None


def test_old_variable_still_works(monkeypatch):
    """An existing .env must not break on the rename."""
    from pipeline.config import load_config
    from pipeline.llm import get_llm
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert get_llm(load_config(), "repo_analysis") is not None


def test_error_names_the_new_variable(monkeypatch):
    from pipeline.config import load_config
    from pipeline.llm import get_llm
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        get_llm(load_config(), "repo_analysis")


# ------------------------------------------- uncapped tiers (NVIDIA's actual one)
def test_zero_budget_means_uncapped_not_exhausted(cfg):
    """`0` is a deliberate 'no cap', not an empty wallet. A falsy-check here
    would have flipped it back into a cap - which it once did."""
    cfg["llm"]["free_tier_credits"] = 0
    led = credits.load(cfg)
    assert not led.capped
    assert not led.exhausted


def test_uncapped_summary_reports_usage_without_a_ceiling(cfg):
    cfg["llm"]["free_tier_credits"] = 0
    led = credits.load(cfg)
    for _ in range(300):
        led.record(stage="s", run_id="r", model="m")
    out = led.summary()
    assert "300 calls made" in out
    assert "remaining" not in out and "full runs" not in out


def test_uncapped_tier_never_warns(cfg):
    cfg["llm"]["free_tier_credits"] = 0
    led = credits.load(cfg)
    for _ in range(5000):
        led.record(stage="s", run_id="r", model="m")
    led.save()
    assert credits.preflight_check(cfg, 138, "A full pipeline run") == ""


def test_a_capped_provider_still_warns(cfg):
    """Switching provider back to one that meters calls must restore the guard."""
    cfg["llm"]["free_tier_credits"] = 100
    led = credits.load(cfg)
    for _ in range(95):
        led.record(stage="s", run_id="r", model="m")
    led.save()
    assert "only 5 remain" in credits.preflight_check(cfg, 138, "A full pipeline run")
