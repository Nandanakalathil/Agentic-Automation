"""Maps the standard endpoint schema <-> the model adapter.

This is the ONLY file tied to the shape of schema/endpoint_schema.json.
When you replace the schema, update these three functions (once, for all
models); per-model adapters keep working because they only see
`inputs` (list) and `parameters` (dict) and return `outputs` (list).
"""
from __future__ import annotations

import uuid
from typing import Any


def parse_request(body: dict) -> tuple[str, list, dict]:
    """Return (request_id, inputs, parameters) from a schema-valid request."""
    return body.get("request_id") or str(uuid.uuid4()), body["inputs"], body.get("parameters") or {}


def build_response(request_id: str, outputs: list, *, model_name: str, model_version: str,
                   latency_ms: float) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "model": {"name": model_name, "version": model_version},
        "outputs": outputs,
        "latency_ms": round(latency_ms, 2),
    }


def build_error(request_id: str, code: str, message: str) -> dict[str, Any]:
    """code is one of: INVALID_REQUEST, MODEL_ERROR, TIMEOUT, INTERNAL."""
    return {"request_id": request_id, "error": {"code": code, "message": message[:2000]}}


HTTP_STATUS = {"INVALID_REQUEST": 400, "MODEL_ERROR": 500, "TIMEOUT": 504, "INTERNAL": 500}
