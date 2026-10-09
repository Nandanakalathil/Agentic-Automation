"""LLM factory: OpenRouter through the OpenAI-compatible chat API."""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from .config import Config
from .credits import CreditTracker


def get_llm(cfg: Config, stage: str, run_id: str = "") -> ChatOpenAI:
    """Return the chat model configured for a pipeline stage (see llm.models)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env or the environment).")

    llm_cfg = cfg["llm"]
    model = llm_cfg["models"].get(stage)
    if not model:
        raise KeyError(f"No model configured for stage '{stage}' in llm.models")

    # Sent via extra_body: ChatOpenAI renames max_tokens to max_completion_tokens,
    # which OpenRouter does not document; max_tokens is its normalised parameter.
    extra_body = {"max_tokens": llm_cfg.get("max_output_tokens", 8192)}
    if llm_cfg.get("provider_preferences"):
        extra_body["provider"] = llm_cfg["provider_preferences"]

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=llm_cfg["base_url"],
        temperature=llm_cfg.get("temperature", 0.1),
        timeout=llm_cfg.get("request_timeout_s", 300),
        max_retries=3,
        extra_body=extra_body,
        default_headers={"X-Title": "model-deploy-pipeline"},
        # Attached here, not in the nodes, so every call is counted - including the
        # JSON-repair retries inside to_structured, which no node knows about.
        callbacks=[CreditTracker(cfg, stage=stage, run_id=run_id, model=model)],
    )
