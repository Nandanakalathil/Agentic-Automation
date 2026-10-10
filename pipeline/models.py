"""Structured outputs each agent must return (validated before the graph moves on)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ analysis
class FileRole(BaseModel):
    path: str
    role: str


class Link(BaseModel):
    kind: Literal["github", "huggingface", "arxiv", "paper", "docs", "colab", "website", "other"] = "other"
    url: str
    title: str = ""


class Licensing(BaseModel):
    """Code and weights are separate permissions; serving needs both."""
    code: str = Field("", description="licence of the source code, e.g. Apache-2.0")
    code_file: str = ""
    weights: str = Field("", description="licence/terms of the model weights - often stricter than the code")
    weights_source: str = Field("", description="where the weights terms are stated: file, model card, paper")
    commercial_use: Literal["yes", "no", "unclear"] = "unclear"
    attribution_required: bool = False
    notes: str = ""


class Capability(BaseModel):
    """One thing a user can actually do with this repo."""
    task: str = Field(description="e.g. 'monomer structure prediction', 'inverse folding'")
    entrypoint: str = Field("", description="script or function that performs it")
    inputs: str = ""
    outputs: str = ""
    supported: Literal["yes", "partial", "needs_extra_assets", "no"] = "yes"
    notes: str = ""


class Script(BaseModel):
    path: str
    kind: Literal["cli", "library", "notebook", "shell", "config", "test", "other"] = "other"
    purpose: str = ""
    invocation: str = Field("", description="exact command line, e.g. 'python run.py --fasta x.fa'")
    key_args: list[str] = []


class Asset(BaseModel):
    path: str
    kind: Literal["weights", "config", "data", "example", "docs", "docker", "notebook", "other"] = "other"
    size_mb: float | None = None
    description: str = ""
    bundled: bool = Field(True, description="False when it must be downloaded separately")


class IOSpec(BaseModel):
    """One input or output of the model's main inference path."""
    name: str
    format: str = Field("", description="fasta, pdb, mmcif, a3m, smiles, json, image, tensor, ...")
    required: bool = True
    description: str = ""
    constraints: str = Field("", description="e.g. 'max 2048 residues', 'single chain only'")
    example: str = ""


class FlowStep(BaseModel):
    """One node of the process-flow diagram. Rendered deterministically, so the
    model supplies structure rather than diagram syntax it tends to get wrong."""
    id: str = Field(description="short slug, no spaces, e.g. 'msa_search'")
    label: str
    kind: Literal["input", "process", "model", "output", "decision", "external"] = "process"
    next: list[str] = Field([], description="ids of the following steps")


class Alert(BaseModel):
    """Something the operator must see before using or deploying this repo."""
    severity: Literal["blocker", "warning", "info"] = "warning"
    title: str
    detail: str = ""


class Quickstart(BaseModel):
    """The shortest honest path from a clean machine to one real prediction."""
    install: list[str] = Field([], description="exact shell commands, in order")
    get_weights: list[str] = Field([], description="commands or manual steps to obtain weights")
    run: str = Field("", description="one command that produces a prediction, using a real example input")
    example_input: str = Field("", description="a file in the repo, or a literal value that works")
    expected_output: str = Field("", description="what success looks like: file written, shape, value range")
    time_estimate: str = ""
    caveats: list[str] = Field([], description="what will bite someone on a first attempt")


class Metadata(BaseModel):
    model_name: str = ""
    owner: str = Field("", description="organisation or author")
    status: Literal["active", "maintained", "archived", "deprecated", "unknown"] = "unknown"
    purpose: str = Field("", description="one sentence: what it is for")
    category: str = Field("", description="e.g. structure prediction, sequence design, property prediction")
    details: str = Field("", description="2-4 sentences of substance for a first-time reader")
    publication: str = Field("", description="paper title, venue and year if stated")
    model_size: str = Field("", description="parameter count or checkpoint size, e.g. '650M params / 2.6 GB'")
    links: list[Link] = []


class FunctionInfo(BaseModel):
    file: str
    name: str
    signature: str = ""
    purpose: str = ""


class Weights(BaseModel):
    source: str = Field("", description="e.g. HuggingFace hub id, URL, bundled in repo, trained by script")
    location_or_id: str = ""
    approx_size_gb: float | None = None
    download_instructions: str = ""


class Hardware(BaseModel):
    gpu_required: bool = False
    cpu_only_supported: bool = Field(False, description="True if it genuinely runs without a GPU")
    min_gpu_memory_gb: float | None = None
    peak_memory_gb: float | None = Field(None, description="observed/documented peak GPU or RAM use at inference")
    peak_memory_basis: str = Field("", description="where that figure comes from: README, code, issue, estimate")
    min_ram_gb: float | None = None
    disk_gb: float | None = Field(None, description="weights plus any databases the model needs on disk")
    approx_runtime: str = Field("", description="e.g. '~3 min for a 300-residue protein on an A10G'")
    notes: str = ""


class TestInfo(BaseModel):
    framework: str = Field("", description="pytest, unittest, custom script, none")
    paths: list[str] = []
    command: str = ""
    approx_count: int | None = None
    needs_gpu: bool = False
    needs_weights: bool = False
    # A repo with no test suite may still ship a known input and its committed
    # output. That pair is a regression check, and the only correctness evidence
    # available before deployment - so it is worth as much as a test suite here.
    reference_input: str = Field("", description="bundled example input file, if any")
    reference_output: str = Field("", description="committed expected output for that input")
    reference_note: str = Field("", description="what to compare, e.g. 'E0VIU9 top EC should be 2.3.2.31'")


class InferenceEntrypoint(BaseModel):
    file: str = ""
    function_or_class: str = ""
    how_to_call: str = ""
    input_format: str = ""
    output_format: str = ""


class RepoAnalysis(BaseModel):
    summary: str
    task: str = Field(description="what the model does, e.g. 'text classification', 'image segmentation'")
    frameworks: list[str] = []
    python_version: str = ""
    dependency_files: list[str] = []
    notable_dependencies: list[str] = []
    key_files: list[FileRole] = []
    functions: list[FunctionInfo] = []
    inference: InferenceEntrypoint = InferenceEntrypoint()
    weights: Weights = Weights()
    hardware: Hardware = Hardware()
    tests: TestInfo = TestInfo()
    license: str = ""
    risks: list[str] = []
    # --- orientation for a first-time reader -----------------------------
    metadata: Metadata = Metadata()
    quickstart: Quickstart = Quickstart()
    licensing: Licensing = Licensing()
    capabilities: list[Capability] = Field([], description="every task this repo can actually perform")
    scripts: list[Script] = Field([], description="runnable scripts and what each is for")
    assets: list[Asset] = Field([], description="weights, configs, example data, notebooks")
    inputs: list[IOSpec] = Field([], description="inputs of the main inference path")
    outputs: list[IOSpec] = Field([], description="outputs of the main inference path")
    flow: list[FlowStep] = Field([], description="process flow of the repo's main pipeline, as a graph")
    alerts: list[Alert] = Field([], description="what an operator must know before using this repo")


# ------------------------------------------------------------------ EC2
class Ec2Plan(BaseModel):
    instance_type: str
    rationale: str
    root_volume_gb: int = 150
    setup_commands: list[str] = Field(default_factory=list, description="environment setup, run in repo dir")
    weights_commands: list[str] = []
    test_command: str = ""
    smoke_inference_command: str = ""
    expected_duration_min: int = 60
    risks: list[str] = []


class TestTotals(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0


class TestFailure(BaseModel):
    test: str
    reason: str
    resolved: bool = False


class FixApplied(BaseModel):
    file_or_command: str
    change: str
    reason: str


class Ec2TestReport(BaseModel):
    status: Literal["passed", "partial", "failed", "no_tests"]
    test_command: str = ""
    totals: TestTotals = TestTotals()
    failures: list[TestFailure] = []
    fixes_applied: list[FixApplied] = []
    environment: dict[str, str] = Field(default_factory=dict, description="python, cuda, gpu, key package versions")
    smoke_inference: str = Field("", description="command run + short result, or why not run")
    model_artifact_dir: str | None = Field(None, description="absolute path on EC2 of downloaded weights, if any")
    notes: str = ""


# ------------------------------------------------------------------ SageMaker
class AutoscalingPlan(BaseModel):
    min_capacity: int = 1
    max_capacity: int = 4
    target_invocations_per_instance: int = 50


class SageMakerPlan(BaseModel):
    instance_type: str
    rationale: str
    initial_instance_count: int = 1
    base_image: str = Field(description="Docker base image for serving, e.g. pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    weights_strategy: Literal["model_tar_gz", "baked_into_image", "download_at_startup"] = "model_tar_gz"
    model_artifact_dir: str | None = Field(None, description="EC2 dir packaged to model.tar.gz (if model_tar_gz)")
    max_concurrency: int = 1
    environment: dict[str, str] = {}
    expected_latency_ms: float | None = None
    autoscaling: AutoscalingPlan = AutoscalingPlan()
    risks: list[str] = []


class AdapterResult(BaseModel):
    files_written: list[str] = []
    contract_passed: bool
    contract_summary: str = ""
    notes: str = ""
