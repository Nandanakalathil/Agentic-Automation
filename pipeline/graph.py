"""The pipeline graph.

setup -> analyze -> plan_ec2 -> [approve_ec2] -> provision_ec2 -> run_ec2_tests (retry loop)
      -> ec2_test_summary -> plan_sagemaker -> [approve_sagemaker] -> build_adapter (retry loop)
      -> deploy_endpoint -> configure_scaling -> test_endpoint -> finalize
Any failure past its retry budget, or a rejection, routes to finalize (which tears down EC2).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import Config
from .nodes import Nodes
from .state import PipelineState


def build_graph(cfg: Config, checkpointer=None):
    n = Nodes(cfg)
    max_fix = cfg["agent"]["max_fix_attempts"]

    def after_approval(stage: str, replan: str, proceed: str):
        def route(s: PipelineState) -> str:
            decision = s.get(f"{stage}_decision")
            if decision == "approve":
                return proceed
            if decision == "revise":
                return replan
            return "finalize"
        return route

    def after_ec2_tests(s: PipelineState) -> str:
        failed = s["ec2_test_results"]["status"] == "failed"
        return "run_ec2_tests" if failed and s["test_fix_attempts"] < max_fix else "ec2_test_summary"

    def adapter_ok(s: PipelineState) -> bool:
        return bool((s.get("adapter_results") or {}).get("verified_contract_passed"))

    def retry_adapter_or(next_ok: str):
        def route(s: PipelineState) -> str:
            if adapter_ok(s):
                return next_ok
            return "build_adapter" if s.get("adapter_fix_attempts", 0) < max_fix else "finalize"
        return route

    def after_endpoint_tests(s: PipelineState) -> str:
        if s["endpoint_test_results"].get("passed"):
            return "finalize"
        return "build_adapter" if s.get("adapter_fix_attempts", 0) < max_fix else "finalize"

    g = StateGraph(PipelineState)
    for name in ["setup_run", "analyze_repo", "plan_ec2", "approve_ec2", "provision_ec2", "run_ec2_tests",
                 "ec2_test_summary", "plan_sagemaker", "approve_sagemaker", "build_adapter", "deploy_endpoint",
                 "configure_scaling", "test_endpoint", "finalize"]:
        g.add_node(name, getattr(n, name))

    g.add_edge(START, "setup_run")
    g.add_edge("setup_run", "analyze_repo")
    g.add_edge("analyze_repo", "plan_ec2")
    g.add_edge("plan_ec2", "approve_ec2")
    g.add_conditional_edges("approve_ec2", after_approval("ec2", "plan_ec2", "provision_ec2"),
                            ["plan_ec2", "provision_ec2", "finalize"])
    g.add_edge("provision_ec2", "run_ec2_tests")
    g.add_conditional_edges("run_ec2_tests", after_ec2_tests, ["run_ec2_tests", "ec2_test_summary"])
    g.add_edge("ec2_test_summary", "plan_sagemaker")
    g.add_edge("plan_sagemaker", "approve_sagemaker")
    g.add_conditional_edges("approve_sagemaker", after_approval("sagemaker", "plan_sagemaker", "build_adapter"),
                            ["plan_sagemaker", "build_adapter", "finalize"])
    g.add_conditional_edges("build_adapter", retry_adapter_or("deploy_endpoint"),
                            ["deploy_endpoint", "build_adapter", "finalize"])
    g.add_conditional_edges("deploy_endpoint", retry_adapter_or("configure_scaling"),
                            ["configure_scaling", "build_adapter", "finalize"])
    g.add_edge("configure_scaling", "test_endpoint")
    g.add_conditional_edges("test_endpoint", after_endpoint_tests, ["finalize", "build_adapter"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
