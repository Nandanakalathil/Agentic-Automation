"""Prompt templates format cleanly and report renderers handle realistic + sparse data."""
import json

from pipeline import prompts, reports
from pipeline.config import load_config
from pipeline.models import (AdapterResult, Ec2Plan, Ec2TestReport, RepoAnalysis,
                             SageMakerPlan)

STATE = {"run_id": "20261005-x", "repo_url": "https://github.com/o/r", "ec2_instance_id": "i-1",
         "ec2_plan": {"instance_type": "g5.xlarge"}, "sagemaker_plan": {"instance_type": "ml.g5.xlarge"},
         "endpoint_name": "mdp-r-x", "autoscaling": {"min": 1, "max": 4, "alarms": ["a"]}}


def test_prompts_format_without_key_errors():
    cfg = load_config()
    prompts.EC2_PLANNER.format(allowed="g5.xlarge", max_hourly=6, feedback="")
    prompts.EC2_TESTER.format(plan="{}", analysis="{}", previous="")
    prompts.SAGEMAKER_PLANNER.format(allowed="x", max_hourly=8, min_cap=1, max_cap=4,
                                     schema=json.dumps(cfg.load_schema()), ec2_report="{}", analysis="{}", feedback="")
    out = prompts.ADAPTER_BUILDER.format(plan="{}", ec2_report="{}", analysis="{}", schema="{}", previous="")
    assert '{"valid":' in out
    for m in (RepoAnalysis, Ec2Plan, Ec2TestReport, SageMakerPlan, AdapterResult):
        assert "```json" in prompts.output_contract(m)


def test_renderers_with_minimal_models():
    a = RepoAnalysis(summary="s", task="t").model_dump()
    assert "Repository Analysis" in reports.repo_analysis(STATE, a, "narr\n```json\n{\"a\": 1}\n```")
    p = Ec2Plan(instance_type="g5.xlarge", rationale="r").model_dump()
    assert "~$1.01/h" in reports.ec2_plan(STATE, p, 1.01, "")
    assert "unknown" in reports.ec2_plan(STATE, p, None, "")
    r = Ec2TestReport(status="partial").model_dump()
    assert "PARTIAL" in reports.ec2_test_summary(STATE, r, "")
    sp = SageMakerPlan(instance_type="ml.g5.xlarge", rationale="r", base_image="img").model_dump()
    md = reports.sagemaker_plan(STATE, sp, 1.41, "")
    assert "$1,029" in md and "$4,117" in md  # 1.41 * 730 * {1, 4}
    results = {"contract": {"total": 1, "passed": 1, "failed": 0,
                            "cases": [{"name": "c", "kind": "valid", "passed": True, "detail": "", "latency_ms": 12.3}]},
               "load": {"requests": 10, "concurrency": 2, "error_rate_pct": 0, "throughput_rps": 5,
                        "latency_ms": {"p50": 1, "p95": 2, "p99": 3}}}
    assert "PASSED" in reports.endpoint_test_summary(STATE, results)
    guide = reports.integration_guide(STATE, load_config().load_schema(), {"inputs": ["hi"]}, "us-east-1")
    assert "sagemaker:InvokeEndpoint" in guide and "mdp-r-x" in guide
    assert "Pipeline Run Report" in reports.final_report({**STATE, "status": "completed", "artifacts": {"x": "/a/x.md"}})


def test_narrative_strips_trailing_json():
    assert reports.narrative("hello\n\n```json\n{\"k\": {\"n\": 1}}\n```\n") == "hello"
