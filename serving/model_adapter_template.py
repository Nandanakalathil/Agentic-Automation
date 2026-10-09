"""Model adapter - the ONLY model-specific serving code.

The adapter_builder agent rewrites this file for each repo. Contract:
  - MODEL_NAME / MODEL_VERSION: strings reported in every response.
  - load(model_dir): load weights once. model_dir is /opt/ml/model (from
    model.tar.gz) - weights may also be baked into the image under /opt/program.
  - predict(model, inputs, parameters): `inputs` is a list (batch); return a list
    of JSON-serialisable outputs, one per input, same order.
    Raise ValueError for invalid user input -> HTTP 400.
"""
from __future__ import annotations

MODEL_NAME = "example-model"
MODEL_VERSION = "0.0.0"


def load(model_dir: str):
    return {"loaded_from": model_dir}


def predict(model, inputs: list, parameters: dict) -> list:
    for x in inputs:
        if not isinstance(x, str):
            raise ValueError("each input must be a string")
    return [{"echo": x, "length": len(x)} for x in inputs]
