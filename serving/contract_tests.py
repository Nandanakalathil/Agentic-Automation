"""Schema-contract + load tests for a standard endpoint.

Standalone (stdlib + jsonschema) so it can run on the EC2 box against the
local container, and be imported by the pipeline for SageMaker endpoints.

  python contract_tests.py --url http://localhost:8080 \
      --schema endpoint_schema.json --samples sample_requests.json

sample_requests.json: {"valid": [<request>, ...], "invalid": [<request>, ...],
                       "golden": [<golden case>, ...]}
("invalid" is optional; generic invalid cases are generated and kept only if
the schema itself rejects them, so this works for any schema.)

A golden case pins the model's actual numbers, not just its HTTP status:

  {"name": "reference sequence", "request": {"inputs": ["MKTAYIAKQRQISFVK"]},
   "checks": [{"path": "outputs", "len": 1},
              {"path": "outputs.0.mean_plddt", "min": 70, "max": 100}]}

Each check selects a dotted path into the response and asserts one of
`equals`, `len`, `contains`, or a `min`/`max` numeric range. "golden" is
optional - omit it and nothing changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from jsonschema import Draft202012Validator

Send = Callable[[Any], "tuple[int, Any, float]"]


def http_transport(base_url: str, timeout: float = 120) -> tuple[Send, Callable[[], int]]:
    def send(payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"{base_url}/invocations", data=data,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        except OSError as e:  # connection refused/reset, timeout
            return 0, f"transport error: {e}", (time.perf_counter() - t0) * 1000
        ms = (time.perf_counter() - t0) * 1000
        try:
            return status, json.loads(raw), ms
        except json.JSONDecodeError:
            return status, raw.decode(errors="replace"), ms

    def ping():
        try:
            with urllib.request.urlopen(f"{base_url}/ping", timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError:
            return 0

    return send, ping


def _validator(schema: dict, name: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": schema["$defs"]})


MAX_TEST_PAYLOAD_BYTES = 5 * 1024 * 1024  # stay under SageMaker's 6 MB limit (enforced before the container)


def _generic_invalid(valid: dict, req_v: Draft202012Validator, req_schema: dict) -> list[tuple[str, Any]]:
    cands: list[tuple[str, Any]] = [("empty object", {}), ("array body", []), ("string body", "hello")]
    props = req_schema.get("properties", {}) if isinstance(req_schema, dict) else {}
    if isinstance(valid, dict):
        extra = copy.deepcopy(valid)
        extra["__unexpected_field__"] = 1
        cands.append(("unknown top-level field", extra))
        for k, v in valid.items():
            missing = {kk: vv for kk, vv in valid.items() if kk != k}
            cands.append((f"missing '{k}'", missing))
            if isinstance(v, list) and v:
                cands.append((f"'{k}' empty list", {**valid, k: []}))
                cands.append((f"'{k}' wrong type", {**valid, k: "not-a-list"}))
                max_items = props.get(k, {}).get("maxItems")
                if isinstance(max_items, int):  # only when the schema defines a limit
                    over = {**valid, k: (v * (max_items // len(v) + 1))[: max_items + 1]}
                    if len(json.dumps(over)) <= MAX_TEST_PAYLOAD_BYTES:
                        cands.append((f"'{k}' over maxItems ({max_items})", over))
    return [(n, c) for n, c in cands if not req_v.is_valid(c)]


def run_contract_tests(send: Send, ping: Callable[[], int] | None, schema: dict, samples: dict) -> dict:
    req_v, resp_v, err_v = (_validator(schema, n) for n in ("request", "response", "error"))
    cases: list[dict] = []

    def record(name, kind, ok, detail="", ms=None):
        cases.append({"name": name, "kind": kind, "passed": bool(ok), "detail": detail[:500], "latency_ms": ms})

    if ping:
        st = ping()
        record("GET /ping", "health", st == 200, f"status {st}")

    valid = samples.get("valid") or []
    for i, req in enumerate(valid):
        errs = [e.message for e in req_v.iter_errors(req)]
        if errs:
            record(f"valid[{i}] sample is schema-valid", "sample", False, "; ".join(errs[:3]))
            continue
        status, body, ms = send(req)
        problems = []
        if status != 200:
            problems.append(f"status {status}: {str(body)[:200]}")
        elif not isinstance(body, dict):
            problems.append("response is not a JSON object")
        else:
            problems += [f"schema: {e.message}" for e in list(resp_v.iter_errors(body))[:3]]
            if isinstance(req, dict) and req.get("request_id") and body.get("request_id") != req["request_id"]:
                problems.append("request_id not echoed")
            if isinstance(req, dict) and isinstance(req.get("inputs"), list) and isinstance(body.get("outputs"), list) \
                    and len(req["inputs"]) != len(body["outputs"]):
                problems.append(f"{len(req['inputs'])} inputs but {len(body['outputs'])} outputs")
        record(f"valid[{i}]", "valid", not problems, "; ".join(problems), ms)

    invalid = [(f"invalid[{i}]", r) for i, r in enumerate(samples.get("invalid") or [])]
    if valid:
        invalid += _generic_invalid(valid[0], req_v, schema["$defs"]["request"])
    for name, req in invalid:
        status, body, ms = send(req)
        problems = []
        if not 400 <= status < 500:
            problems.append(f"expected 4xx, got {status}")
        if not isinstance(body, dict) or not err_v.is_valid(body):
            problems.append(f"error body does not match schema: {str(body)[:200]}")
        record(name, "invalid", not problems, "; ".join(problems), ms)

    for i, g in enumerate(samples.get("golden") or []):
        _run_golden(send, record, i, g)

    passed = sum(c["passed"] for c in cases)
    return {"total": len(cases), "passed": passed, "failed": len(cases) - passed, "cases": cases}


def _select(obj: Any, path: str) -> tuple[bool, Any]:
    """Walk a dotted path like 'outputs.0.mean_plddt'. Returns (found, value)."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        elif isinstance(cur, list):
            if not part.lstrip("-").isdigit() or not -len(cur) <= int(part) < len(cur):
                return False, None
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def _check_golden(body: Any, check: dict) -> str:
    """Return an empty string when the check passes, else why it failed."""
    path = check.get("path", "")
    found, val = (True, body) if path in ("", ".") else _select(body, path)
    if not found:
        return f"{path}: not present in response"
    if "equals" in check and val != check["equals"]:
        return f"{path}: expected {check['equals']!r}, got {val!r}"
    if "len" in check:
        if not isinstance(val, (list, str, dict)):
            return f"{path}: len check on non-sized {type(val).__name__}"
        if len(val) != check["len"]:
            return f"{path}: expected length {check['len']}, got {len(val)}"
    if "contains" in check and check["contains"] not in str(val):
        return f"{path}: {check['contains']!r} not in {str(val)[:120]!r}"
    if "min" in check or "max" in check:
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return f"{path}: expected a number, got {type(val).__name__} ({str(val)[:60]})"
        if "min" in check and val < check["min"]:
            return f"{path}: {val} < min {check['min']}"
        if "max" in check and val > check["max"]:
            return f"{path}: {val} > max {check['max']}"
    return ""


def _run_golden(send: Send, record, i: int, g: dict) -> None:
    """A known input whose output must land in a known range.

    A container that builds and returns HTTP 200 is not proof of a correct model:
    a wrong dtype, a missing weights file or a silently-swapped checkpoint all
    still return 200. These cases assert on the numbers themselves, so silent
    regressions fail the deployment instead of shipping.
    """
    name = f"golden[{i}] {g.get('name', '')}".strip()
    req = g.get("request")
    if not isinstance(req, dict):
        record(name, "golden", False, "golden case has no 'request' object")
        return
    status, body, ms = send(req)
    if status != 200:
        record(name, "golden", False, f"status {status}: {str(body)[:200]}", ms)
        return
    failures = [msg for msg in (_check_golden(body, c) for c in g.get("checks") or []) if msg]
    record(name, "golden", not failures, "; ".join(failures[:3]), ms)


def run_load_test(send: Send, request: dict, *, concurrency: int, total: int) -> dict:
    def one(_):
        status, _body, ms = send(request)
        return status, ms

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(total)))
    wall = time.perf_counter() - t0
    lat = sorted(ms for st, ms in results if st == 200)
    errors = sum(1 for st, _ in results if st != 200)

    def pct(p):
        return round(lat[min(len(lat) - 1, int(p / 100 * len(lat)))], 1) if lat else None

    return {
        "requests": total, "concurrency": concurrency, "errors": errors,
        "error_rate_pct": round(100 * errors / total, 2),
        "throughput_rps": round(total / wall, 2),
        "latency_ms": {"p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99),
                       "mean": round(statistics.mean(lat), 1) if lat else None},
        "status_codes": {str(s): sum(1 for st, _ in results if st == s) for s in {st for st, _ in results}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--wait-ready", type=int, default=900, help="seconds to wait for /ping 200")
    ap.add_argument("--load-requests", type=int, default=50)
    ap.add_argument("--load-concurrency", type=int, default=4)
    a = ap.parse_args()

    send, ping = http_transport(a.url)
    deadline = time.time() + a.wait_ready
    while ping() != 200 and time.time() < deadline:
        time.sleep(5)
    schema = json.load(open(a.schema))
    samples = json.load(open(a.samples))
    out = {"contract": run_contract_tests(send, ping, schema, samples)}
    if samples.get("valid") and a.load_requests > 0:
        out["load"] = run_load_test(send, samples["valid"][0], concurrency=a.load_concurrency, total=a.load_requests)
    print("MDP_RESULT_JSON=" + json.dumps(out))


if __name__ == "__main__":
    main()
