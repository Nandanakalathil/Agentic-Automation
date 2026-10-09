import pytest
from langchain_core.messages import AIMessage, ToolMessage

from pipeline.harness import ELIDED, _elide_old_tool_outputs, parse_structured, run_agent, to_structured
from pipeline.models import Ec2TestReport
from pipeline.tools.repo_tools import make_repo_tools


class ScriptedLLM:
    """Stands in for a chat model: returns queued AIMessages, records what it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def test_agent_loop_runs_tools_and_returns_final(tmp_path):
    (tmp_path / "README.md").write_text("hello model")
    tools = make_repo_tools(tmp_path, writable=False)
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "c1"}]),
        AIMessage("", tool_calls=[{"name": "nope", "args": {}, "id": "c2"}]),
        AIMessage("done"),
    ])
    res = run_agent(llm, name="t", system_prompt="s", task="t", tools=tools, log_dir=tmp_path / "logs")
    assert res.final_text == "done" and res.steps == 3 and not res.hit_step_limit
    tool_msgs = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert "hello model" in tool_msgs[0].content
    assert "unknown tool" in tool_msgs[1].content
    assert (tmp_path / "logs" / "t.jsonl").exists()


def test_step_limit_forces_final_answer(tmp_path):
    tools = make_repo_tools(tmp_path, writable=False)
    loop = AIMessage("", tool_calls=[{"name": "list_dir", "args": {}, "id": "x"}])
    llm = ScriptedLLM([loop, loop, AIMessage("partial result")])
    res = run_agent(llm, name="t", system_prompt="s", task="t", tools=tools, max_steps=2)
    assert res.hit_step_limit and res.final_text == "partial result"


def test_elision_keeps_tool_messages_paired():
    msgs = [AIMessage("", tool_calls=[{"name": "a", "args": {}, "id": str(i)}]) if i % 2 == 0
            else ToolMessage("x" * 1000, tool_call_id=str(i - 1)) for i in range(10)]
    _elide_old_tool_outputs(msgs, budget=2500)
    tms = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tms) == 5 and tms[0].content == ELIDED and tms[-1].content != ELIDED


def test_parse_structured_from_fenced_and_bare_json():
    text = 'Report...\n```json\n{"status": "passed", "totals": {"total": 3, "passed": 3}}\n```'
    r = parse_structured(text, Ec2TestReport)
    assert r.status == "passed" and r.totals.passed == 3
    bare = 'blah {"status": "no_tests", "notes": "brace } inside \\"string\\""} trailing'
    assert parse_structured(bare, Ec2TestReport).notes == 'brace } inside "string"'


def test_to_structured_repairs_via_llm():
    llm = ScriptedLLM([AIMessage('```json\n{"status": "failed"}\n```')])
    r = to_structured(llm, "not json at all", Ec2TestReport)
    assert r.status == "failed"


def test_repo_tools_sandbox(tmp_path):
    tools = {t.name: t for t in make_repo_tools(tmp_path, writable=True)}
    tools["write_file"].invoke({"path": "a/b.py", "content": "x = 1\n"})
    assert (tmp_path / "a" / "b.py").read_text() == "x = 1\n"
    assert "edited" in tools["edit_file"].invoke({"path": "a/b.py", "old_text": "x = 1", "new_text": "x = 2"})
    assert "b.py:1" in tools["search"].invoke({"regex": "x = 2"})
    with pytest.raises(ValueError):
        tools["read_file"].invoke({"path": "../../etc/passwd"})
    ro = {t.name for t in make_repo_tools(tmp_path, writable=False)}
    assert "write_file" not in ro and "edit_file" not in ro
