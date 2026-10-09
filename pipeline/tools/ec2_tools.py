"""Tools that let an agent work on the throwaway EC2 instance."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from ..aws.ec2 import REMOTE_REPO, REMOTE_VENV, Ec2Runner


def make_ec2_tools(runner: Ec2Runner, workspace: Path, max_timeout_s: int = 3600) -> list[BaseTool]:
    def run_on_ec2(command: str, timeout_s: int = 900) -> str:
        r = runner.run(command, timeout_s=min(int(timeout_s), max_timeout_s))
        return json.dumps({"exit_code": r["exit_code"], "status": r.get("status"),
                           "stdout": r["stdout"][-8000:], "stderr": r["stderr"][-4000:]})

    def sync_workspace_to_ec2() -> str:
        """Upload the local repo workspace (including your file edits) to the EC2 instance.
        Call this after write_file/edit_file and before re-running commands."""
        return runner.sync_workspace(workspace)

    run_doc = (f"Run a bash command on the EC2 test instance as root, in {REMOTE_REPO}. "
               f"The python venv at {REMOTE_VENV} is auto-activated. Each call is a fresh shell "
               "(env vars and `cd` do not persist). Returns exit code, stdout and stderr (tail).")
    return [
        StructuredTool.from_function(run_on_ec2, description=run_doc),
        StructuredTool.from_function(sync_workspace_to_ec2),
    ]
