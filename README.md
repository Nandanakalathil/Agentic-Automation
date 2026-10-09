# model-deploy-pipeline

A LangGraph multi-agent pipeline that takes the git URL of an AI model repo and:

1. **Analyses the repo** and writes an orientation document: metadata, capabilities, inputs/outputs, scripts, assets, compute needs, licensing, and a process-flow diagram.
2. **Plans an EC2 deployment**, then pauses for your approval.
3. **Deploys to EC2 and runs the repo's own tests**, fixing environmental failures (never by weakening tests), and writes a test summary.
4. **Plans a SageMaker endpoint**, then pauses for your approval.
5. **Wraps the model in a standard container** that enforces your I/O schema, and contract-tests it on EC2.
6. **Deploys the endpoint** with autoscaling and CloudWatch alarms.
7. **Tests the live endpoint** (schema contract and load test), writes a test summary and an integration guide, and tears down EC2.

Agents run on **open models through NVIDIA NIM**, with the model set per stage in `config/pipeline.yaml`. Any OpenAI-compatible endpoint works — only `llm.base_url` changes.

```
setup → analyze_repo → plan_ec2 → ⏸ approve_ec2 → provision_ec2 → run_ec2_tests ⟲ → ec2_test_summary
      → plan_sagemaker → ⏸ approve_sagemaker → build_adapter ⟲ → deploy_endpoint → configure_scaling
      → test_endpoint → finalize (teardown EC2, reports)
⟲ = bounded retry loop (agent.max_fix_attempts)   ⏸ = human approval (LangGraph interrupt)
```

Every node is checkpointed to `runs/checkpoints.sqlite`, so a crash or Ctrl-C can be resumed from the last completed step.

## Project layout

```
pipeline/              the orchestrator
  graph.py             14 nodes, 2 approval gates, 3 retry loops
  nodes.py             the body of every node
  harness.py           the agent loop (step limits, elision, JSON repair)
  prompts.py           the 5 system prompts + native-build compatibility matrix
  models.py            Pydantic schemas each agent must return
  preflight.py         deterministic repo scan, runs BEFORE the analyst
  verify.py            deterministic checks on the analyst's output, AFTER it
  credits.py           free-tier call ledger and budget warnings
  dev.py               run one agent standalone against saved inputs
  llm.py               model factory (one place; swap providers here)
  config.py state.py reports.py
  aws/                 ec2, sagemaker, pricing, session
  tools/               repo_tools (4 read-only), ec2_tools
serving/               the container every model ships in (serve, envelope, contract tests)
schema/                the endpoint I/O contract
config/pipeline.yaml   models, instance types, cost caps, budgets
tests/                 199 offline tests - no AWS, no LLM calls
runs/<run-id>/         all output, plus credits.json and checkpoints.sqlite
```

Paths are resolved from the package location, so the project runs from any directory.

## Outputs (`runs/<run-id>/`)

| File | Contents |
|---|---|
| `01_repo_analysis.md` / `analysis.json` | Orientation doc: alerts banner, metadata, quickstart, capabilities, process-flow diagram, repo structure, inputs/outputs, scripts, assets, compute, licensing, risks |
| `preflight.json` | The deterministic scan the analyst was given as fact |
| `02_ec2_deployment_plan.md` | Instance choice, cost estimate, setup/weights/test commands |
| `03_ec2_test_summary.md` | Pass/fail counts, failures, **every fix applied** (reproducibility log), environment versions |
| `04_sagemaker_deployment_plan.md` | Instance, base image, weights strategy, monthly cost range, autoscaling |
| `05_sagemaker_test_summary.md` / `endpoint_test_results.json` | Contract cases, p50/p95/p99 latency, throughput, error rate |
| `06_endpoint_integration.md` | What your backend needs: endpoint name, IAM policy, boto3 example, schema |
| `serving_bundle/` | The exact serving files baked into the image |
| `00_final_report.md`, `state.json`, `logs/*.jsonl` | Run summary, final state, full agent transcripts |

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                       # add your API key
AWS_REGION=us-east-1 ./scripts/bootstrap_aws.sh   # one-time: bucket, IAM roles, ECR repo
# put the printed PIPELINE_ARTIFACT_BUCKET into .env
.venv/bin/python -m pytest                 # offline tests, no AWS or LLM calls
```

The API key goes in `OPENROUTER_API_KEY` **whatever the backend is** — `llm.py` reads that
variable name regardless of provider. For NVIDIA NIM, get a key at
[build.nvidia.com](https://build.nvidia.com).

**Permissions the person running the pipeline needs:** EC2 (run/terminate/describe instances), SSM (send-command, get-parameter), S3 on the artifact bucket, ECR, SageMaker (models, endpoint configs, endpoints), Application Auto Scaling, CloudWatch alarms, CloudWatch Logs (read), `iam:GetRole`, `iam:PassRole` on the two pipeline roles, and `iam:CreateServiceLinkedRole` (the first time autoscaling is used).

**GPU quotas:** new AWS accounts often have a quota of 0 vCPUs for G instances. Request "Running On-Demand G and VT instances" (EC2) and `ml.g5.xlarge for endpoint usage` (SageMaker) in Service Quotas before your first GPU run. Check that `aws.sagemaker.autoscaling.max_capacity` does not exceed your endpoint quota — exceeding it fails *after* the endpoint is live and billing.

## Usage

```bash
python -m pipeline run https://github.com/org/model-repo [--ref v1.2]
```

At each approval gate you see the plan report, the cost, and any **deterministic findings** about the analysis the plan rests on (licence blockers, hallucinated paths, broken scripts, contradictions with the scan). Then choose **approve**, **approve with a different instance type**, **revise** (your feedback goes back to the planner), or **reject**.

```bash
python -m pipeline run <url> --no-input            # stops at the gate
python -m pipeline approve <run-id> --decision approve [--instance-type g5.2xlarge]
python -m pipeline approve <run-id> --decision revise --feedback "use CPU, the model is tiny"
python -m pipeline status <run-id>
python -m pipeline resume <run-id>                 # after a crash or Ctrl-C
python -m pipeline list
python -m pipeline cleanup [--run-id ID] [--endpoints] [--yes]   # delete anything left behind
```

### Developing one agent at a time

A full run costs ~138 API calls and, past the first gate, real AWS time. Each agent can be
driven on its own against a previous run's output, so only the agent under test spends credits:

```bash
python -m pipeline agent list                                   # what each needs, and what it costs
python -m pipeline agent repo_analyst --repo https://github.com/org/repo   # ~35 calls, no AWS
python -m pipeline agent ec2_planner       --from-run <run-id>             # ~10 calls, no AWS
python -m pipeline agent sagemaker_planner --from-run <run-id>             # ~10 calls, no AWS
python -m pipeline agent test_fixer      --from-run <run-id> --instance i-0abc
python -m pipeline agent adapter_builder --from-run <run-id> --instance i-0abc
```

`repo_analyst`, `ec2_planner` and `sagemaker_planner` need no AWS at all. `test_fixer` and
`adapter_builder` run commands on a live instance — pass `--instance` to reuse one you already
have, since provisioning a fresh box per iteration is what makes those two expensive.

### Credits

The free tier bills **per API call**, not per token. The pipeline keeps its own ledger and
warns before work that would exhaust it:

```bash
python -m pipeline credits [--reset]
```

A summary prints after every run and every single-agent invocation. Set your real allowance in
`config/pipeline.yaml` (`llm.free_tier_credits`) — the ledger is only as accurate as that number,
and it cannot see calls made outside this pipeline.

## The standard endpoint contract

Every endpoint runs the same serving layer (`serving/`):

| File | Role | Changes |
|---|---|---|
| `schema/endpoint_schema.json` | Your I/O contract (`$defs.request`, `$defs.response`, `$defs.error`) | Once, if you replace it |
| `serving/envelope.py` | Maps the schema to and from `inputs`/`parameters`/`outputs` | Once, together with the schema |
| `serving/serve.py` | HTTP server (`/ping`, `/invocations`) that validates every request **and response** | Never |
| `serving/contract_tests.py` | Schema conformance, golden-output and load tests (generic: works for any schema) | Never |
| `.mdp/model_adapter.py` | `load(model_dir)` and `predict(model, inputs, parameters) -> list` | Written by the agent for each model |

Because the agent only writes the adapter, every endpoint speaks exactly the same contract. A response that doesn't match the schema is turned into an `INTERNAL` error rather than returned to your backend.

The shipped schema is **typed where it matters and permissive everywhere else**: an input or
output carrying a `type` key is validated strictly against a biological definition (protein/DNA/RNA
sequence, PDB/mmCIF structure, A3M alignment, SMILES molecule, with pLDDT/pTM bounds on outputs),
while anything without a `type` key passes through untouched. So a malformed FASTA is rejected at
the edge, and a non-biological model still works unchanged. `schema/endpoint_schema.generic.json`
is the untyped original; switch with `schema_path`.

**Golden cases** pin the model's actual numbers, not just its HTTP status — a wrong checkpoint or
a missing weights file still returns 200. The adapter builder derives them by running the model on
EC2 first.

## Guardrails

- **Approval before anything that bills:** before launching EC2 and before creating the endpoint.
- **Allowed instance types and a $/hour cap** for both EC2 and SageMaker, enforced in code even if a planner or a human picks something else.
- **The analyst's claims are checked, not trusted:** cited file paths must exist, the licence comes from the LICENSE file rather than the model, code and weights licences are read separately, scripts that do not parse are reported, and disagreements with the deterministic scan are surfaced at the approval gate.
- **Credit budget:** the pipeline warns before starting work that would exhaust the free tier, because a run that dies mid-way can strand a billing EC2 instance.
- **EC2 cleanup:** terminated in `finalize`, terminated on OS shutdown, and a **self-destruct timer** set at boot (`self_destruct_hours`, default 24) in case the pipeline dies. If the instance is gone when a later stage needs it (for example, an approval given the next day), a replacement is launched and the approved setup is replayed. Everything is tagged `model-deploy-pipeline=true` for `cleanup`.
- **Approvals can't get stuck:** an invalid answer (an instance type outside the allowed list or over the cost cap, or a garbled decision) brings the prompt back with an error instead of crashing the run.
- **Endpoint cleanup:** a broken endpoint is deleted after the last retry (`delete_endpoint_on_failure`).
- **Agent sandbox:** file tools are confined to the run's workspace, shell commands run only on the throwaway EC2 instance (never on your machine), and dangerous commands are blocked.
- **Results are verified, not trusted:** container contract results come from the test harness, not from what the agent claims.
- **Data:** repo code is sent to the LLM provider. Check this is acceptable for private repos.

## Costs (rough)

- **LLM:** on NVIDIA's free tier, cost is measured in **calls**, not dollars — ~138 per full run against ~1000 free credits, so roughly 7 runs. On a paid per-token provider the same run is ~6.8M input / 65k output tokens; input dominates 100:1.
- **EC2 test instance:** about $1/h on a g5.xlarge, usually 0.5–2 h per run.
- **SageMaker endpoint:** about $1.41/h on ml.g5.xlarge, around **$1,000/month per always-on instance**. A successful run **leaves the endpoint running on purpose** — `finalize` only deletes it on failure. Run `python -m pipeline cleanup --endpoints --yes` when you are done with it.
- Prices in `pipeline/aws/pricing.py` are approximate and will drift. Check current AWS pricing.

## Known limits

- **Not yet run end to end against real AWS.** Request shapes are validated against botocore's API models in the tests, but the first real run is the true test. Start with a small CPU model.
- **Tool-calling support is unverified** for the configured models. NVIDIA's catalog does not publish it. If an agent finishes in one step having read nothing, the model does not support tool calls — swap it in `config/pipeline.yaml`.
- **Open models are less reliable** than frontier models at multi-step debugging. Expect some runs to need a `revise` or a rerun.
- **Large weights (>5 GB)** will strain the `model_tar_gz` strategy; EFS or runtime S3 download is not implemented.
- **Real-time endpoints only:** a 6 MB payload limit and a 60 s per-request limit. Long-running models (video, large generation jobs) need SageMaker Async Inference, which isn't implemented.
- **Private repos:** pass a token-bearing URL or configure git credentials. Git LFS weights are pulled on the instance (`git lfs pull`), not locally.
- **No VPC endpoint configuration and no data capture.** Both can be added in `pipeline/aws/sagemaker.py`.
