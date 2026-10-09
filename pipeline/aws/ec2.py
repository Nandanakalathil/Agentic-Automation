"""EC2 lifecycle + remote command execution through SSM (no SSH keys, no open ports)."""
from __future__ import annotations

import base64
import io
import tarfile
import time
from pathlib import Path

from botocore.exceptions import ClientError

from ..config import Config
from .session import TAG_KEY, boto_session, run_tags

REMOTE_ROOT = "/opt/work"
REMOTE_REPO = f"{REMOTE_ROOT}/repo"
REMOTE_VENV = f"{REMOTE_ROOT}/venv"
SYNC_EXCLUDES = {".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}

# Guardrail: commands the test agent may never run on the instance.
FORBIDDEN_SNIPPETS = ["aws ec2 terminate", "aws iam", "shutdown", "poweroff", "reboot",
                      "rm -rf / ", "rm -rf /*", "mkfs", "aws sagemaker delete"]


class Ec2Runner:
    def __init__(self, cfg: Config, run_id: str, instance_id: str | None = None):
        self.cfg = cfg
        self.run_id = run_id
        self.instance_id = instance_id
        s = boto_session(cfg)
        self.ec2 = s.client("ec2")
        self.ssm = s.client("ssm")
        self.s3 = s.client("s3")
        self.bucket = cfg.get_path("aws.artifact_bucket")
        if not self.bucket:
            raise RuntimeError("aws.artifact_bucket is not set (run scripts/bootstrap_aws.sh).")

    # ------------------------------------------------------------ lifecycle
    def launch(self, instance_type: str, volume_gb: int | None = None) -> str:
        ec2_cfg = self.cfg["aws"]["ec2"]
        ami = self.ssm.get_parameter(Name=ec2_cfg["ami_ssm_parameter"])["Parameter"]["Value"]
        params = dict(
            ImageId=ami,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            IamInstanceProfile={"Name": ec2_cfg["instance_profile"]},
            BlockDeviceMappings=[{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": volume_gb or ec2_cfg["root_volume_gb"], "VolumeType": "gp3",
                        "DeleteOnTermination": True},
            }],
            MetadataOptions={"HttpTokens": "required"},
            # Safety net: if the pipeline dies, `shutdown` from inside terminates the box.
            InstanceInitiatedShutdownBehavior="terminate",
            TagSpecifications=[
                {"ResourceType": "instance",
                 "Tags": run_tags(self.run_id) + [{"Key": "Name", "Value": f"mdp-{self.run_id}"}]},
                {"ResourceType": "volume", "Tags": run_tags(self.run_id)},
            ],
            # Hard TTL in case the pipeline dies: the instance powers off (=> terminates) by itself.
            UserData=f"#!/bin/bash\nshutdown -h +{int(ec2_cfg.get('self_destruct_hours', 24) * 60)}\n",
        )
        if ec2_cfg.get("subnet_id"):
            params["SubnetId"] = ec2_cfg["subnet_id"]
        if ec2_cfg.get("security_group_ids"):
            params["SecurityGroupIds"] = ec2_cfg["security_group_ids"]
        resp = self.ec2.run_instances(**params)
        self.instance_id = resp["Instances"][0]["InstanceId"]
        return self.instance_id

    def wait_until_ready(self, timeout_s: int = 900) -> None:
        self.ec2.get_waiter("instance_running").wait(InstanceIds=[self.instance_id])
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            info = self.ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [self.instance_id]}]
            )["InstanceInformationList"]
            if info and info[0]["PingStatus"] == "Online":
                return
            time.sleep(15)
        raise TimeoutError(f"{self.instance_id} never came online in SSM (check instance profile / outbound internet)")

    def terminate(self) -> None:
        if self.instance_id:
            self.ec2.terminate_instances(InstanceIds=[self.instance_id])

    def is_alive(self) -> bool:
        if not self.instance_id:
            return False
        try:
            r = self.ec2.describe_instances(InstanceIds=[self.instance_id])
        except ClientError:
            return False
        state = r["Reservations"][0]["Instances"][0]["State"]["Name"]
        return state in ("pending", "running")

    # ------------------------------------------------------------ commands
    def run(self, command: str, timeout_s: int | None = None, cwd: str = REMOTE_REPO) -> dict:
        """Run a bash script on the instance as root. Returns exit code + stdout/stderr."""
        lowered = command.lower()
        for bad in FORBIDDEN_SNIPPETS:
            if bad in lowered:
                return {"exit_code": 126, "stdout": "", "stderr": f"blocked by pipeline guardrail: '{bad}'"}

        timeout_s = timeout_s or self.cfg.get_path("aws.ec2.command_timeout_s", 3600)
        script = (
            "set -o pipefail\n"
            "export HOME=/root DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1\n"
            f"mkdir -p {cwd} && cd {cwd}\n"
            f"if [ -f {REMOTE_VENV}/bin/activate ]; then . {REMOTE_VENV}/bin/activate; fi\n"
            f"{command}\n"
        )
        b64 = base64.b64encode(script.encode()).decode()
        prefix = f"runs/{self.run_id}/ssm"
        cmd = self.ssm.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={
                "commands": [f"echo {b64} | base64 -d > /tmp/mdp_cmd.sh", "bash /tmp/mdp_cmd.sh"],
                "executionTimeout": [str(timeout_s)],
            },
            OutputS3BucketName=self.bucket,
            OutputS3KeyPrefix=prefix,
            TimeoutSeconds=600,
        )["Command"]
        cid = cmd["CommandId"]
        deadline = time.time() + timeout_s + 120
        inv = None
        while time.time() < deadline:
            time.sleep(5)
            try:
                inv = self.ssm.get_command_invocation(CommandId=cid, InstanceId=self.instance_id)
            except ClientError as e:
                if "InvocationDoesNotExist" in str(e):
                    continue
                raise
            if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
                break
        if inv is None:
            return {"exit_code": -1, "stdout": "", "stderr": "command never started"}

        # Full output is in S3 (inline content is cut at 24k chars); the upload can lag a few seconds.
        stdout = stderr = ""
        for _ in range(3):
            stdout, stderr = self._s3_output(prefix, cid, "stdout"), self._s3_output(prefix, cid, "stderr")
            if stdout or stderr:
                break
            time.sleep(3)
        stdout = stdout or inv.get("StandardOutputContent", "")
        stderr = stderr or inv.get("StandardErrorContent", "")
        return {
            "exit_code": inv.get("ResponseCode", -1),
            "status": inv["Status"],
            "stdout": stdout,
            "stderr": stderr,
        }

    def _s3_output(self, prefix: str, cid: str, stream: str) -> str:
        # SSM writes one folder per plugin step; ours is the 2nd command (index 1).
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{prefix}/{cid}/{self.instance_id}/")
        parts = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(f"/{stream}")]
        out = []
        for key in sorted(parts):
            out.append(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode(errors="replace"))
        return "".join(out)

    # ------------------------------------------------------------ files
    def sync_workspace(self, local_dir: Path) -> str:
        """Tar the local workspace, upload to S3, and extract it over the remote repo dir."""
        buf = io.BytesIO()

        def _filter(ti: tarfile.TarInfo):
            return None if any(part in SYNC_EXCLUDES for part in Path(ti.name).parts) else ti

        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(local_dir, arcname=".", filter=_filter)
        key = f"runs/{self.run_id}/workspace.tar.gz"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
        r = self.run(
            f"aws s3 cp s3://{self.bucket}/{key} /tmp/ws.tar.gz --only-show-errors && "
            f"tar xzf /tmp/ws.tar.gz -C {REMOTE_REPO} && echo synced",
            timeout_s=900,
        )
        if r["exit_code"] != 0:
            raise RuntimeError(f"workspace sync failed: {r['stderr'][-2000:]}")
        return f"synced {len(buf.getvalue()) // 1024} KiB to {REMOTE_REPO}"

    def bootstrap(self) -> dict:
        """Base tooling: AWS CLI, git, python venv. Idempotent; raises on failure."""
        apt = "apt-get -o DPkg::Lock::Timeout=600 -y -qq"  # unattended-upgrades often holds the lock at boot
        r = self.run(
            "set -e\n"
            f"{apt} update\n"
            f"{apt} install git git-lfs curl python3-venv python3-pip >/dev/null\n"
            f"command -v aws >/dev/null || {apt} install awscli >/dev/null\n"
            # A half-created venv (no bin/activate) would silently never be activated: rebuild it.
            f"[ -f {REMOTE_VENV}/bin/activate ] || {{ rm -rf {REMOTE_VENV}; python3 -m venv {REMOTE_VENV}; }}\n"
            f"{REMOTE_VENV}/bin/python -m pip install -q --upgrade pip jsonschema\n"
            "nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'no GPU'\n"
            "python3 --version; docker --version",
            cwd=REMOTE_ROOT,
            timeout_s=1800,
        )
        if r["exit_code"] != 0:
            raise RuntimeError(f"EC2 bootstrap failed (exit {r['exit_code']}):\n{r['stdout'][-3000:]}\n{r['stderr'][-3000:]}")
        return r


def find_tagged_instances(cfg: Config) -> list[dict]:
    ec2 = boto_session(cfg).client("ec2")
    resp = ec2.describe_instances(Filters=[
        {"Name": f"tag:{TAG_KEY}", "Values": ["true"]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopped"]},
    ])
    return [i for r in resp["Reservations"] for i in r["Instances"]]
