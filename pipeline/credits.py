"""Credit ledger for the free tier.

NVIDIA's developer tier bills per API CALL, not per token: roughly 1000 credits,
one credit per call. A full pipeline run makes ~138 calls, so the free tier is
about seven runs - close enough to the ceiling that crossing it by accident is
easy, and a run that dies mid-way because credits ran out can strand a running
EC2 instance.

So every call is counted here, as it happens, against a configured budget. The
count comes from a LangChain callback rather than from arithmetic in the nodes,
which means it catches calls the nodes do not know about - notably the JSON
repair retries inside `to_structured`.

The ledger is the pipeline's own record of what IT spent. It cannot see calls
made from anywhere else, and the 1000 figure is from NVIDIA's published free
tier, not from your account - check the dashboard for the authoritative number.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_LOCK = threading.Lock()
LEDGER_NAME = "credits.json"


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def to_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens}


@dataclass
class Ledger:
    path: Path
    budget: int = 1000
    total: Usage = field(default_factory=Usage)
    by_stage: dict[str, dict] = field(default_factory=dict)
    by_run: dict[str, dict] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    # -------------------------------------------------------------- storage
    @classmethod
    def load(cls, path: Path, budget: int = 1000) -> "Ledger":
        led = cls(path=path, budget=budget)
        if path.exists():
            try:
                d = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return led
            t = d.get("total") or {}
            led.total = Usage(t.get("calls", 0), t.get("input_tokens", 0), t.get("output_tokens", 0))
            led.by_stage = d.get("by_stage") or {}
            led.by_run = d.get("by_run") or {}
            led.history = d.get("history") or []
        return led

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "budget": self.budget,
            "total": self.total.to_dict(),
            "by_stage": self.by_stage,
            "by_run": self.by_run,
            "history": self.history[-200:],
        }, indent=2))

    # ---------------------------------------------------------------- state
    @property
    def capped(self) -> bool:
        """False when the provider has no total cap, only a rate limit.

        NVIDIA's free tier works this way: `Up to 40 rpm` and no credit balance.
        The ledger still counts calls - knowing a run cost 138 is useful - but
        there is nothing to run out of, so it must not warn as though there is.
        """
        return self.budget > 0

    @property
    def used(self) -> int:
        return self.total.calls

    @property
    def remaining(self) -> int:
        return max(self.budget - self.used, 0) if self.capped else 0

    @property
    def exhausted(self) -> bool:
        return self.capped and self.remaining <= 0

    def record(self, *, stage: str, run_id: str, model: str,
               input_tokens: int = 0, output_tokens: int = 0) -> None:
        u = Usage(1, input_tokens, output_tokens)
        self.total.add(u)
        for bucket, key in ((self.by_stage, stage or "unknown"), (self.by_run, run_id or "adhoc")):
            cur = bucket.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            cur["calls"] += 1
            cur["input_tokens"] += input_tokens
            cur["output_tokens"] += output_tokens
        if model and (not self.history or self.history[-1].get("model") != model
                      or self.history[-1].get("run_id") != run_id):
            self.history.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "run_id": run_id, "stage": stage, "model": model})

    # --------------------------------------------------------------- output
    def summary(self) -> str:
        if not self.capped:
            lines = ["", "  API usage  (no credit cap on this tier - rate limited only)",
                     f"  {self.used} calls made so far"]
            if self.total.input_tokens:
                lines.append(f"  tokens: {self.total.input_tokens / 1e6:.2f}M in / "
                             f"{self.total.output_tokens / 1000:.1f}k out")
            return "\n".join(lines)

        pct = 100 * self.used / self.budget
        bar_len = 28
        filled = min(int(bar_len * pct / 100), bar_len)
        bar = "=" * filled + "-" * (bar_len - filled)
        lines = [
            "",
            "  NVIDIA free-tier credits",
            f"  [{bar}] {self.used}/{self.budget} used  ({pct:.0f}%)",
            f"  remaining: {self.remaining} calls  (~{self.remaining // 138} full runs, "
            f"~{self.remaining // 35} analyst runs)",
        ]
        if self.total.input_tokens:
            lines.append(f"  tokens: {self.total.input_tokens / 1e6:.2f}M in / "
                         f"{self.total.output_tokens / 1000:.1f}k out")
        return "\n".join(lines)

    def per_run_summary(self, run_id: str) -> str:
        r = self.by_run.get(run_id)
        if not r:
            return ""
        return (f"  this run: {r['calls']} calls"
                + (f", {r['input_tokens'] / 1e6:.2f}M input tokens" if r["input_tokens"] else ""))


# ---------------------------------------------------------------- integration
def ledger_path(cfg) -> Path:
    return Path(cfg.runs_dir) / LEDGER_NAME


def load(cfg) -> Ledger:
    # `or 1000` would turn a deliberate 0 (uncapped) back into a cap.
    budget = cfg.get_path("llm.free_tier_credits", 0)
    return Ledger.load(ledger_path(cfg), int(budget) if budget is not None else 0)


class CreditTracker(BaseCallbackHandler):
    """Counts one credit per completed model call, straight from LangChain.

    Hooked at the model rather than the node so that retries, repair passes and
    anything else that reaches the API is counted too - the ledger should match
    the provider's number, not the pipeline's intentions.
    """

    def __init__(self, cfg, stage: str, run_id: str = "", model: str = ""):
        self.cfg, self.stage, self.run_id, self.model = cfg, stage, run_id, model

    def on_llm_end(self, response: Any, **_kw: Any) -> None:
        try:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            in_tok = int(usage.get("prompt_tokens") or 0)
            out_tok = int(usage.get("completion_tokens") or 0)
            with _LOCK:
                led = load(self.cfg)
                led.record(stage=self.stage, run_id=self.run_id, model=self.model,
                           input_tokens=in_tok, output_tokens=out_tok)
                led.save()
                if led.remaining and led.remaining <= int(self.cfg.get_path("llm.warn_at_remaining", 150)):
                    print(f"\n  LOW CREDITS: {led.remaining} calls left of {led.budget}. "
                          f"A full run needs ~138.\n")
                elif led.exhausted:
                    print("\n  FREE TIER EXHAUSTED by the pipeline's own count. "
                          "Check build.nvidia.com before starting anything that provisions AWS.\n")
        except Exception:  # accounting must never break a run
            pass


def preflight_check(cfg, planned_calls: int, what: str) -> str:
    """Warning text to show BEFORE starting work that might exhaust the tier. '' when fine."""
    led = load(cfg)
    if not led.capped:          # nothing to exhaust; a rate limit is not a budget
        return ""
    if led.exhausted:
        return (f"Credits exhausted ({led.used}/{led.budget} by this pipeline's count). "
                f"{what} needs ~{planned_calls} more. Verify at build.nvidia.com before continuing.")
    if planned_calls > led.remaining:
        return (f"{what} needs ~{planned_calls} calls but only {led.remaining} remain. "
                f"It will likely stop part-way. "
                f"{'An unfinished run can leave an EC2 instance billing - run `cleanup` if it dies.' if planned_calls > 100 else ''}")
    if planned_calls > led.remaining * 0.5:
        return (f"{what} needs ~{planned_calls} calls; {led.remaining} remain. "
                f"This will use over half of what is left.")
    return ""


# Rough call counts per agent, from config limits (see dev.py / the step budgets).
PLANNED_CALLS = {
    "repo_analyst": 35, "ec2_planner": 10, "test_fixer": 40,
    "sagemaker_planner": 10, "adapter_builder": 40, "full_run": 138,
}
