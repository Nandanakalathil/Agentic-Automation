from __future__ import annotations

import boto3

from ..config import Config

TAG_KEY = "model-deploy-pipeline"


def boto_session(cfg: Config) -> boto3.Session:
    return boto3.Session(profile_name=cfg.get_path("aws.profile"), region_name=cfg.get_path("aws.region"))


def run_tags(run_id: str) -> list[dict[str, str]]:
    return [{"Key": TAG_KEY, "Value": "true"}, {"Key": "pipeline-run-id", "Value": run_id}]


def account_id(session: boto3.Session) -> str:
    return session.client("sts").get_caller_identity()["Account"]
