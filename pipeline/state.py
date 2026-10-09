"""Shared state passed between LangGraph nodes.

Everything here is JSON-serialisable so the SQLite checkpointer can persist it
and a crashed run can resume from the last completed node.
"""
from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    # Inputs
    run_id: str
    repo_url: str
    repo_ref: str | None
    run_dir: str          # runs/<run_id>
    workspace: str        # runs/<run_id>/repo (local clone the agents edit)

    # Stage 1: repo analysis
    analysis: dict[str, Any]

    # Stage 2-3: EC2
    ec2_plan: dict[str, Any]
    ec2_approved: bool
    ec2_instance_id: str | None
    ec2_test_results: dict[str, Any]
    test_fix_attempts: int

    # Stage 4-5: SageMaker
    sagemaker_plan: dict[str, Any]
    sagemaker_approved: bool
    adapter_results: dict[str, Any]
    adapter_fix_attempts: int
    image_uri: str | None
    endpoint_name: str | None
    autoscaling: dict[str, Any]
    endpoint_test_results: dict[str, Any]

    # Human feedback from approval gates (sent back to the planner on "revise")
    ec2_decision: str | None        # approve | revise | reject
    sagemaker_decision: str | None
    ec2_feedback: str | None
    sagemaker_feedback: str | None
    last_failure: str | None   # context handed to the next retry of a fix loop
    commit_sha: str | None
    alarms: list[str]

    # Bookkeeping
    status: str           # running | rejected | failed | completed
    errors: list[str]
    artifacts: dict[str, str]   # name -> path of generated reports
