"""Generic SageMaker-compatible inference server (identical for every model).

GET  /ping         -> 200 once the model is loaded, 503 while loading/failed
POST /invocations  -> standard schema request/response (see envelope.py)

Model-specific code lives only in model_adapter.py:
    MODEL_NAME, MODEL_VERSION
    load(model_dir) -> model
    predict(model, inputs: list, parameters: dict) -> list   (same length as inputs)
Raise ValueError in predict() for bad user input (-> 400 INVALID_REQUEST).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import envelope  # noqa: E402
import model_adapter  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serve")

SCHEMA = json.loads((HERE / "endpoint_schema.json").read_text())
REQ_VALIDATOR = Draft202012Validator({"$ref": "#/$defs/request", "$defs": SCHEMA["$defs"]})
RESP_VALIDATOR = Draft202012Validator({"$ref": "#/$defs/response", "$defs": SCHEMA["$defs"]})

MODEL_DIR = os.environ.get("MDP_MODEL_DIR", "/opt/ml/model")
MAX_BODY = int(os.environ.get("MDP_MAX_BODY_BYTES", str(6 * 1024 * 1024)))  # SageMaker realtime limit
PREDICT_SLOTS = threading.BoundedSemaphore(int(os.environ.get("MDP_MAX_CONCURRENCY", "1")))
STRICT_RESPONSES = os.environ.get("MDP_STRICT_RESPONSES", "1") == "1"

_state = {"model": None, "ready": False, "error": None}


def _load():
    t0 = time.time()
    try:
        _state["model"] = model_adapter.load(MODEL_DIR)
        _state["ready"] = True
        log.info("model loaded in %.1fs", time.time() - t0)
    except Exception:
        _state["error"] = traceback.format_exc()
        log.error("model load failed:\n%s", _state["error"])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # route access logs through logging
        log.debug(fmt, *args)

    def _send(self, status: int, payload: dict | str):
        body = (json.dumps(payload) if isinstance(payload, dict) else payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ping":
            if _state["ready"]:
                return self._send(200, {"status": "ok"})
            return self._send(503, {"status": "failed" if _state["error"] else "loading"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/invocations":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            # Drain (bounded) so the client isn't reset mid-send, then close so leftover
            # bytes are never parsed as the next request.
            remaining = min(length, 64 * 1024 * 1024)
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            return self._error("unknown", "INVALID_REQUEST", f"body exceeds {MAX_BODY} bytes")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"null")
        except json.JSONDecodeError as e:
            return self._error("unknown", "INVALID_REQUEST", f"invalid JSON: {e}")

        rid = body.get("request_id") if isinstance(body, dict) and isinstance(body.get("request_id"), str) else "unknown"
        errors = sorted(REQ_VALIDATOR.iter_errors(body), key=lambda e: list(e.path))
        if errors:
            msg = "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5])
            return self._error(rid, "INVALID_REQUEST", msg)
        if not _state["ready"]:
            return self._error(rid, "INTERNAL", "model not loaded")

        request_id, inputs, params = envelope.parse_request(body)
        t0 = time.perf_counter()
        try:
            with PREDICT_SLOTS:
                outputs = model_adapter.predict(_state["model"], inputs, params)
        except ValueError as e:
            return self._error(request_id, "INVALID_REQUEST", str(e))
        except Exception as e:
            log.error("predict failed:\n%s", traceback.format_exc())
            return self._error(request_id, "MODEL_ERROR", f"{type(e).__name__}: {e}")
        latency = (time.perf_counter() - t0) * 1000

        if not isinstance(outputs, list) or len(outputs) != len(inputs):
            return self._error(request_id, "INTERNAL",
                               f"adapter returned {type(outputs).__name__} of wrong length (expected {len(inputs)})")
        resp = envelope.build_response(request_id, outputs, model_name=model_adapter.MODEL_NAME,
                                       model_version=model_adapter.MODEL_VERSION, latency_ms=latency)
        resp_errors = list(RESP_VALIDATOR.iter_errors(resp))
        if resp_errors:
            log.error("response violates schema: %s", resp_errors[0].message)
            if STRICT_RESPONSES:
                return self._error(request_id, "INTERNAL", f"response violates schema: {resp_errors[0].message}")
        self._send(200, resp)

    def _error(self, rid: str, code: str, message: str):
        self._send(envelope.HTTP_STATUS.get(code, 500), envelope.build_error(rid, code, message))


def main():
    threading.Thread(target=_load, daemon=True).start()
    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))
    log.info("listening on :%d", port)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":  # SageMaker runs `docker run <image> serve`; extra args are ignored
    main()
