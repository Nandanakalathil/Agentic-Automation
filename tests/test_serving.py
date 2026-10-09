"""Runs the real serve.py (with the template adapter) and the real contract tests against it."""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVING = ROOT / "serving"
sys.path.insert(0, str(SERVING))
import contract_tests  # noqa: E402


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("mdp")
    for f in ["serve.py", "envelope.py"]:
        shutil.copy(SERVING / f, d / f)
    shutil.copy(SERVING / "model_adapter_template.py", d / "model_adapter.py")
    shutil.copy(ROOT / "schema" / "endpoint_schema.json", d / "endpoint_schema.json")
    port = _free_port()
    env = {**os.environ, "SAGEMAKER_BIND_TO_PORT": str(port), "MDP_MODEL_DIR": str(d)}
    proc = subprocess.Popen([sys.executable, str(d / "serve.py"), "serve"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    _, ping = contract_tests.http_transport(url)
    for _ in range(50):
        if ping() == 200:
            break
        time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait(5)


SAMPLES = {
    "valid": [
        {"request_id": "r-1", "inputs": ["hello world"]},
        {"inputs": ["a", "bb", "ccc"], "parameters": {"top_k": 3}},
    ],
    "invalid": [{"request_id": "r-bad", "inputs": [123]}],  # schema-valid, adapter must reject
}


def test_contract_suite_passes_against_template(server):
    send, ping = contract_tests.http_transport(server)
    schema = json.loads((ROOT / "schema" / "endpoint_schema.json").read_text())
    result = contract_tests.run_contract_tests(send, ping, schema, SAMPLES)
    failed = [c for c in result["cases"] if not c["passed"]]
    assert not failed, failed
    assert result["total"] >= 8  # health + 2 valid + explicit invalid + generated invalid cases


def test_response_envelope(server):
    send, _ = contract_tests.http_transport(server)
    status, body, _ = send({"request_id": "abc", "inputs": ["xy"]})
    assert status == 200
    assert body["request_id"] == "abc"
    assert body["model"] == {"name": "example-model", "version": "0.0.0"}
    assert body["outputs"] == [{"echo": "xy", "length": 2}]


def test_error_envelope_for_bad_json_and_bad_input(server):
    send, _ = contract_tests.http_transport(server)
    status, body, _ = send({"inputs": []})
    assert status == 400 and body["error"]["code"] == "INVALID_REQUEST"
    status, body, _ = send({"request_id": "z", "inputs": [5]})
    assert status == 400 and body["request_id"] == "z" and "string" in body["error"]["message"]


def test_load_test(server):
    send, _ = contract_tests.http_transport(server)
    r = contract_tests.run_load_test(send, SAMPLES["valid"][0], concurrency=4, total=40)
    assert r["errors"] == 0 and r["latency_ms"]["p95"] is not None


def test_large_inputs_pass_contract(server):
    """Regression: the generated oversized case used to exceed 6 MB and reset the connection."""
    send, ping = contract_tests.http_transport(server)
    schema = json.loads((ROOT / "schema" / "endpoint_schema.json").read_text())
    big = {"valid": [{"inputs": ["x" * 60000]}]}  # ~60 KB item, like a base64 image
    result = contract_tests.run_contract_tests(send, ping, schema, big)
    failed = [c for c in result["cases"] if not c["passed"]]
    assert not failed, failed
    assert any("maxItems" in c["name"] for c in result["cases"])


def test_body_over_limit_is_rejected_cleanly(server):
    import urllib.request
    import urllib.error
    data = json.dumps({"inputs": ["y" * (7 * 1024 * 1024)]}).encode()
    req = urllib.request.Request(f"{server}/invocations", data=data, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 400 and json.loads(e.value.read())["error"]["code"] == "INVALID_REQUEST"
    send, _ = contract_tests.http_transport(server)
    assert send({"inputs": ["still alive"]})[0] == 200


def test_contract_detects_broken_adapter(server):
    """A server returning the wrong output count must fail the contract."""
    def bad_send(payload):
        status, body, ms = contract_tests.http_transport(server)[0](payload)
        if status == 200:
            body = {**body, "outputs": []}
        return status, body, ms
    schema = json.loads((ROOT / "schema" / "endpoint_schema.json").read_text())
    result = contract_tests.run_contract_tests(bad_send, None, schema, SAMPLES)
    assert result["failed"] >= 1
