"""Load config/pipeline.yaml into a plain dict with a few helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


class Config(dict):
    """dict subclass with dotted lookup: cfg.get_path("aws.ec2.max_hourly_usd")."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def runs_dir(self) -> Path:
        return PROJECT_ROOT / self.get("runs_dir", "runs")

    def load_schema(self) -> dict:
        return json.loads((PROJECT_ROOT / self["schema_path"]).read_text())


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config" / "pipeline.yaml"
    cfg = Config(yaml.safe_load(path.read_text()))
    # Env overrides for the values most often changed per machine.
    if os.environ.get("AWS_REGION"):
        cfg["aws"]["region"] = os.environ["AWS_REGION"]
    if os.environ.get("PIPELINE_ARTIFACT_BUCKET"):
        cfg["aws"]["artifact_bucket"] = os.environ["PIPELINE_ARTIFACT_BUCKET"]
    return cfg
