"""Minimal, model-agnostic agent harness.

Deliberately not using LangGraph's prebuilt agents: the loop is ~100 lines,
stable across library versions, and gives us exactly the controls open models
need - step limits, tool-output truncation, context elision, transcript logs,
and a validated JSON result.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence, Type, TypeVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

ELIDED = "[older tool output elided to save context]"


class AgentResult(BaseModel):
    final_text: str
    steps: int
    hit_step_limit: bool


def _msg_chars(m: BaseMessage) -> int:
    c = m.content if isinstance(m.content, str) else json.dumps(m.content)
    extra = json.dumps(getattr(m, "tool_calls", None) or [])
    return len(c) + len(extra)


def _elide_old_tool_outputs(messages: list[BaseMessage], budget: int) -> None:
    """Replace the oldest tool outputs with a stub until under budget.

    ToolMessages are kept (not deleted) so every tool_call id still has a reply,
    which OpenAI-compatible APIs require.
    """
    total = sum(_msg_chars(m) for m in messages)
    for m in messages:
        if total <= budget:
            return
        if isinstance(m, ToolMessage) and m.content != ELIDED:
            total -= len(m.content) - len(ELIDED)
            m.content = ELIDED


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n\n... [{len(text) - limit} chars truncated] ...\n\n{text[-tail:]}"


def run_agent(
    llm,
    *,
    name: str,
    system_prompt: str,
    task: str,
    tools: Sequence[BaseTool],
    max_steps: int = 40,
    max_tool_output_chars: int = 12000,
    context_char_budget: int = 400000,
    log_dir: Path | None = None,
    on_step: Callable[[int, AIMessage], None] | None = None,
) -> AgentResult:
    """Run a tool-calling loop until the model answers without tool calls."""
    tool_map = {t.name: t for t in tools}
    bound = llm.bind_tools(list(tools)) if tools else llm
    messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(task)]
    log_path = (log_dir / f"{name}.jsonl") if log_dir else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(event: dict[str, Any]) -> None:
        if log_path:
            event["ts"] = time.time()
            with log_path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")

    log({"event": "start", "task": task[:2000]})
    for step in range(1, max_steps + 1):
        _elide_old_tool_outputs(messages, context_char_budget)
        ai: AIMessage = bound.invoke(messages)
        messages.append(ai)
        log({"event": "ai", "step": step, "content": ai.content, "tool_calls": ai.tool_calls})
        if on_step:
            on_step(step, ai)

        if not ai.tool_calls:
            return AgentResult(final_text=_text(ai), steps=step, hit_step_limit=False)

        for call in ai.tool_calls:
            tool = tool_map.get(call["name"])
            if tool is None:
                output = f"ERROR: unknown tool '{call['name']}'. Available: {sorted(tool_map)}"
            else:
                try:
                    output = tool.invoke(call["args"])
                except Exception as e:  # tool errors go back to the model, not up the stack
                    output = f"ERROR: {type(e).__name__}: {e}"
            output = _truncate(str(output), max_tool_output_chars)
            log({"event": "tool", "name": call["name"], "args": call["args"], "output": output[:4000]})
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    # Step limit hit: ask for a final answer with tools disabled.
    messages.append(HumanMessage(
        "You have reached the step limit. Stop using tools and give your final answer now, "
        "in the exact format requested, including what remains unfinished."
    ))
    _elide_old_tool_outputs(messages, context_char_budget)
    ai = llm.invoke(messages)
    log({"event": "final_after_limit", "content": ai.content})
    return AgentResult(final_text=_text(ai), steps=max_steps, hit_step_limit=True)


def _text(ai: AIMessage) -> str:
    if isinstance(ai.content, str):
        return ai.content
    return "".join(p.get("text", "") for p in ai.content if isinstance(p, dict))


# ---------------------------------------------------------------- JSON output

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def _first_json_object(text: str) -> str | None:
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_structured(text: str, model_cls: Type[T]) -> T:
    raw = _first_json_object(text)
    if raw is None:
        raise ValueError("no JSON object found in model output")
    return model_cls.model_validate(json.loads(raw))


def to_structured(llm, text: str, model_cls: Type[T], retries: int = 2) -> T:
    """Parse `text` into `model_cls`; if that fails, ask the LLM to reformat it.

    Open models are inconsistent with native structured-output modes, so we use
    plain JSON + validation + repair, which works on any chat model.
    """
    try:
        return parse_structured(text, model_cls)
    except (ValueError, ValidationError, json.JSONDecodeError) as first_err:
        err: Exception = first_err
    schema = json.dumps(model_cls.model_json_schema(), indent=1)
    for _ in range(retries):
        prompt = (
            "Convert the following content into ONE JSON object that validates against this JSON Schema. "
            "Output only the JSON inside a ```json fence. Use null or empty lists for unknown values; "
            f"do not invent facts.\n\nSchema:\n{schema}\n\nPrevious error: {err}\n\nContent:\n{text[:60000]}"
        )
        out = _text(llm.invoke([HumanMessage(prompt)]))
        try:
            return parse_structured(out, model_cls)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            err = e
    raise ValueError(f"could not get valid {model_cls.__name__} JSON from model: {err}")
