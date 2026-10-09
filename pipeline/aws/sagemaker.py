"""SageMaker endpoint lifecycle: image build (on the EC2 box), deploy, autoscaling, alarms."""
from __future__ import annotations

import json
import re
import time

from botocore.exceptions import ClientError

from ..config import Config
from .ec2 import REMOTE_REPO, Ec2Runner
from .session import account_id, boto_session, run_tags


def endpoint_name_for(repo_slug: str, run_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", repo_slug).strip("-").lower()[:36].strip("-")
    suffix = re.sub(r"[^a-zA-Z0-9]", "", run_id)[-8:]
    return f"mdp-{slug}-{suffix}"  # <= 49 chars, leaves room for "-model" etc.


class SageMakerDeployer:
    def __init__(self, cfg: Config, run_id: str):
        self.cfg = cfg
        self.run_id = run_id
        s = boto_session(cfg)
        self.region = s.region_name
        self.account = account_id(s)
        self.sm = s.client("sagemaker")
        self.smr = s.client("sagemaker-runtime")
        self.ecr = s.client("ecr")
        self.iam = s.client("iam")
        self.aas = s.client("application-autoscaling")
        self.cw = s.client("cloudwatch")
        self.sm_cfg = cfg["aws"]["sagemaker"]

    # ------------------------------------------------------------ image
    def ensure_ecr_repo(self) -> str:
        name = self.sm_cfg["ecr_repository"]
        try:
            self.ecr.create_repository(repositoryName=name, imageScanningConfiguration={"scanOnPush": True})
        except ClientError as e:
            if "RepositoryAlreadyExists" not in str(e):
                raise
        return f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/{name}"

    def push_local_image(self, runner: Ec2Runner, local_image: str, tag: str) -> str:
        """Push the image already built + contract-tested on the EC2 instance to ECR."""
        repo_uri = self.ensure_ecr_repo()
        image = f"{repo_uri}:{tag}"
        registry = repo_uri.split("/")[0]
        r = runner.run(
            f"aws ecr get-login-password --region {self.region} | docker login --username AWS --password-stdin {registry} && "
            f"docker tag {local_image} {image} && docker push {image} 2>&1 | tail -n 5",
            timeout_s=3600,
            cwd=REMOTE_REPO,
        )
        if r["exit_code"] != 0:
            raise RuntimeError(f"image push failed:\n{r['stdout'][-4000:]}\n{r['stderr'][-2000:]}")
        return image

    def package_model_artifacts(self, runner: Ec2Runner, remote_dir: str | None) -> str | None:
        """tar a directory on the instance into s3://.../model.tar.gz (mounted at /opt/ml/model)."""
        if not remote_dir:
            return None
        key = f"runs/{self.run_id}/model/model.tar.gz"
        uri = f"s3://{runner.bucket}/{key}"
        r = runner.run(f"tar czf /tmp/model.tar.gz -C {remote_dir} . && aws s3 cp /tmp/model.tar.gz {uri} --only-show-errors",
                       timeout_s=3600)
        if r["exit_code"] != 0:
            raise RuntimeError(f"model artifact packaging failed: {r['stderr'][-2000:]}")
        return uri

    # ------------------------------------------------------------ deploy
    def execution_role_arn(self) -> str:
        return self.iam.get_role(RoleName=self.sm_cfg["execution_role_name"])["Role"]["Arn"]

    def deploy(self, *, endpoint_name: str, image_uri: str, model_data_url: str | None,
               instance_type: str, initial_count: int = 1, env: dict | None = None,
               startup_timeout_s: int = 1200) -> dict:
        tags = run_tags(self.run_id)
        container = {"Image": image_uri, "Environment": {k: str(v) for k, v in (env or {}).items()}}
        if model_data_url:
            container["ModelDataUrl"] = model_data_url
        model_name = f"{endpoint_name}-model"
        config_name = f"{endpoint_name[:50]}-{int(time.time())}"  # SageMaker names max 63 chars
        self.sm.create_model(ModelName=model_name, PrimaryContainer=container,
                             ExecutionRoleArn=self.execution_role_arn(), Tags=tags)
        self.sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": initial_count,
                "ContainerStartupHealthCheckTimeoutInSeconds": startup_timeout_s,
                "ModelDataDownloadTimeoutInSeconds": 1800,
            }],
            Tags=tags,
        )
        self.sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=tags)
        return self.wait_in_service(endpoint_name)

    def wait_in_service(self, endpoint_name: str, timeout_s: int = 3600) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            d = self.sm.describe_endpoint(EndpointName=endpoint_name)
            if d["EndpointStatus"] == "InService":
                return {"status": "InService", "endpoint_arn": d["EndpointArn"]}
            if d["EndpointStatus"] == "Failed":
                return {"status": "Failed", "reason": d.get("FailureReason", "unknown"),
                        "logs": self.recent_logs(endpoint_name)}
            time.sleep(30)
        return {"status": "Timeout"}

    def recent_logs(self, endpoint_name: str, limit: int = 100) -> str:
        logs = boto_session(self.cfg).client("logs")
        group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
        try:
            streams = logs.describe_log_streams(logGroupName=group, orderBy="LastEventTime",
                                                descending=True, limit=3)["logStreams"]
            lines = []
            for st in streams:
                ev = logs.get_log_events(logGroupName=group, logStreamName=st["logStreamName"],
                                         limit=limit, startFromHead=False)["events"]
                lines += [e["message"] for e in ev]
            return "\n".join(lines[-limit:])
        except ClientError as e:
            return f"(no logs: {e})"

    # ------------------------------------------------------------ scaling + alarms
    def configure_autoscaling(self, endpoint_name: str, *, min_capacity: int, max_capacity: int,
                              target_invocations: int, scale_in_cooldown: int, scale_out_cooldown: int) -> dict:
        resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
        dim = "sagemaker:variant:DesiredInstanceCount"
        self.aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=resource_id,
                                          ScalableDimension=dim, MinCapacity=min_capacity,
                                          MaxCapacity=max_capacity)
        policy = self.aas.put_scaling_policy(
            PolicyName=f"{endpoint_name}-invocations-target",
            ServiceNamespace="sagemaker", ResourceId=resource_id, ScalableDimension=dim,
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": float(target_invocations),
                "PredefinedMetricSpecification": {"PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"},
                "ScaleInCooldown": scale_in_cooldown,
                "ScaleOutCooldown": scale_out_cooldown,
            },
        )
        return {"resource_id": resource_id, "min": min_capacity, "max": max_capacity,
                "target_invocations_per_instance_per_min": target_invocations,
                "policy_arn": policy["PolicyARN"]}

    def create_alarms(self, endpoint_name: str, *, p95_latency_ms: float, error_rate_pct: float,
                      sns_topic_arn: str | None) -> list[str]:
        dims = [{"Name": "EndpointName", "Value": endpoint_name}, {"Name": "VariantName", "Value": "AllTraffic"}]
        actions = {"AlarmActions": [sns_topic_arn], "OKActions": [sns_topic_arn]} if sns_topic_arn else {}
        names = []
        name = f"{endpoint_name}-p95-latency"
        self.cw.put_metric_alarm(
            AlarmName=name, Namespace="AWS/SageMaker", MetricName="ModelLatency", Dimensions=dims,
            ExtendedStatistic="p95", Period=300, EvaluationPeriods=3,
            Threshold=p95_latency_ms * 1000,  # ModelLatency is in microseconds
            ComparisonOperator="GreaterThanThreshold", TreatMissingData="notBreaching", **actions)
        names.append(name)
        name = f"{endpoint_name}-5xx-rate"
        self.cw.put_metric_alarm(
            AlarmName=name, EvaluationPeriods=3, ComparisonOperator="GreaterThanThreshold",
            Threshold=error_rate_pct, TreatMissingData="notBreaching",
            Metrics=[
                {"Id": "errors", "ReturnData": False, "MetricStat": {"Metric": {
                    "Namespace": "AWS/SageMaker", "MetricName": "Invocation5XXErrors", "Dimensions": dims},
                    "Period": 300, "Stat": "Sum"}},
                {"Id": "calls", "ReturnData": False, "MetricStat": {"Metric": {
                    "Namespace": "AWS/SageMaker", "MetricName": "Invocations", "Dimensions": dims},
                    "Period": 300, "Stat": "Sum"}},
                {"Id": "rate", "Expression": "IF(calls > 0, 100 * errors / calls, 0)", "Label": "5xx %",
                 "ReturnData": True},
            ], **actions)
        names.append(name)
        return names

    # ------------------------------------------------------------ invoke / cleanup
    def invoke(self, endpoint_name: str, payload: dict, timeout_retry: bool = True) -> tuple[int, dict | str, float]:
        t0 = time.perf_counter()
        try:
            r = self.smr.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json",
                                         Accept="application/json", Body=json.dumps(payload))
            body = r["Body"].read().decode()
            status = 200
        except ClientError as e:
            # ModelError carries the container's status code + body.
            err = e.response.get("Error", {})
            status = e.response.get("OriginalStatusCode") or e.response["ResponseMetadata"].get("HTTPStatusCode", 500)
            body = e.response.get("OriginalMessage") or err.get("Message", "")
        ms = (time.perf_counter() - t0) * 1000
        try:
            return status, json.loads(body), ms
        except (json.JSONDecodeError, TypeError):
            return status, body, ms

    def delete_endpoint(self, endpoint_name: str) -> None:
        try:
            cfg_name = self.sm.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]
        except ClientError:
            cfg_name = None
        try:
            self.aas.deregister_scalable_target(ServiceNamespace="sagemaker",
                                                ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
                                                ScalableDimension="sagemaker:variant:DesiredInstanceCount")
        except ClientError:
            pass
        for fn, kw in [(self.sm.delete_endpoint, {"EndpointName": endpoint_name}),
                       (self.sm.delete_endpoint_config, {"EndpointConfigName": cfg_name}),
                       (self.sm.delete_model, {"ModelName": f"{endpoint_name}-model"})]:
            if all(kw.values()):
                try:
                    fn(**kw)
                except ClientError:
                    pass
        if cfg_name:  # deletion is async; a same-name create would fail until it finishes
            self.sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name,
                                                        WaiterConfig={"Delay": 15, "MaxAttempts": 80})
        self.cw.delete_alarms(AlarmNames=[f"{endpoint_name}-p95-latency", f"{endpoint_name}-5xx-rate"])
