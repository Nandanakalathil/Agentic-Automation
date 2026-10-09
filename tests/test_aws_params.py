"""Validate every AWS request the pipeline makes against botocore's API models (no AWS calls)."""
import boto3
import pytest
from botocore.stub import ANY, Stubber

from pipeline.aws import ec2 as ec2_mod
from pipeline.aws.sagemaker import SageMakerDeployer, endpoint_name_for
from pipeline.config import load_config


def _session():
    return boto3.Session(aws_access_key_id="x", aws_secret_access_key="y", region_name="us-east-1")


@pytest.fixture
def cfg():
    c = load_config()
    c["aws"]["artifact_bucket"] = "bucket"
    return c


def test_endpoint_name_limits():
    name = endpoint_name_for("A_Very-Long_Repository.Name-" * 4, "20261005-143000-some-repo")
    assert len(name) <= 49 and name.startswith("mdp-") and "--" not in name
    assert len(f"{name}-model") <= 63 and len(f"{name[:50]}-1790000000") <= 63


def test_ec2_launch_request_is_valid(cfg, monkeypatch):
    monkeypatch.setattr(ec2_mod, "boto_session", lambda c: _session())
    r = ec2_mod.Ec2Runner(cfg, "run-1")
    with Stubber(r.ssm) as ssm, Stubber(r.ec2) as ec2:
        ssm.add_response("get_parameter", {"Parameter": {"Value": "ami-123"}},
                         {"Name": cfg["aws"]["ec2"]["ami_ssm_parameter"]})
        ec2.add_response("run_instances", {"Instances": [{"InstanceId": "i-abc"}]}, {
            "ImageId": "ami-123", "InstanceType": "g5.xlarge", "MinCount": 1, "MaxCount": 1,
            "IamInstanceProfile": {"Name": "model-pipeline-ec2-role"}, "BlockDeviceMappings": ANY,
            "MetadataOptions": {"HttpTokens": "required"}, "InstanceInitiatedShutdownBehavior": "terminate",
            "TagSpecifications": ANY, "UserData": ANY})
        assert r.launch("g5.xlarge") == "i-abc"


def test_ssm_send_command_request_is_valid(cfg, monkeypatch):
    monkeypatch.setattr(ec2_mod, "boto_session", lambda c: _session())
    monkeypatch.setattr(ec2_mod.time, "sleep", lambda s: None)
    r = ec2_mod.Ec2Runner(cfg, "run-1", "i-abc")
    with Stubber(r.ssm) as ssm, Stubber(r.s3) as s3:
        ssm.add_response("send_command", {"Command": {"CommandId": "12345678-1234-1234-1234-123456789012"}}, {
            "InstanceIds": ["i-abc"], "DocumentName": "AWS-RunShellScript", "Parameters": ANY,
            "OutputS3BucketName": "bucket", "OutputS3KeyPrefix": "runs/run-1/ssm", "TimeoutSeconds": 600})
        ssm.add_response("get_command_invocation", {"Status": "Success", "ResponseCode": 0,
                                                    "StandardOutputContent": "inline-out"},
                         {"CommandId": "12345678-1234-1234-1234-123456789012", "InstanceId": "i-abc"})
        for _ in range(6):
            s3.add_response("list_objects_v2", {"Contents": []}, None)
        out = r.run("echo hi")
    assert out == {"exit_code": 0, "status": "Success", "stdout": "inline-out", "stderr": ""}


def test_guardrail_blocks_dangerous_commands(cfg, monkeypatch):
    monkeypatch.setattr(ec2_mod, "boto_session", lambda c: _session())
    r = ec2_mod.Ec2Runner(cfg, "run-1", "i-abc")
    assert r.run("sudo shutdown -h now")["exit_code"] == 126
    assert r.run("aws iam create-user --user-name x")["exit_code"] == 126


def _deployer(cfg):
    d = SageMakerDeployer.__new__(SageMakerDeployer)
    s = _session()
    d.cfg, d.run_id, d.region, d.account = cfg, "run-1", "us-east-1", "123456789012"
    d.sm, d.iam = s.client("sagemaker"), s.client("iam")
    d.aas, d.cw = s.client("application-autoscaling"), s.client("cloudwatch")
    d.sm_cfg = cfg["aws"]["sagemaker"]
    return d


def test_sagemaker_deploy_requests_are_valid(cfg, monkeypatch):
    d = _deployer(cfg)
    monkeypatch.setattr("pipeline.aws.sagemaker.time.sleep", lambda s: None)
    role = "arn:aws:iam::123456789012:role/model-pipeline-sagemaker-role"
    with Stubber(d.sm) as sm, Stubber(d.iam) as iam:
        iam.add_response("get_role", {"Role": {"Path": "/", "RoleName": "r", "RoleId": "AROAEXAMPLEID123456",
                                               "Arn": role, "CreateDate": "2026-01-01T00:00:00Z",
                                               "AssumeRolePolicyDocument": "{}"}},
                         {"RoleName": "model-pipeline-sagemaker-role"})
        sm.add_response("create_model", {"ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/m"}, {
            "ModelName": "mdp-x-model", "ExecutionRoleArn": role, "Tags": ANY,
            "PrimaryContainer": {"Image": "img:1", "ModelDataUrl": "s3://b/model.tar.gz",
                                 "Environment": {"MDP_MAX_CONCURRENCY": "2"}}})
        sm.add_response("create_endpoint_config", {"EndpointConfigArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint-config/c"},
                        {"EndpointConfigName": ANY, "ProductionVariants": ANY, "Tags": ANY})
        sm.add_response("create_endpoint", {"EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/mdp-x"},
                        {"EndpointName": "mdp-x", "EndpointConfigName": ANY, "Tags": ANY})
        sm.add_response("describe_endpoint", {
            "EndpointName": "mdp-x", "EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/mdp-x",
            "EndpointConfigName": "c", "EndpointStatus": "InService",
            "CreationTime": "2026-01-01T00:00:00Z", "LastModifiedTime": "2026-01-01T00:00:00Z"},
            {"EndpointName": "mdp-x"})
        out = d.deploy(endpoint_name="mdp-x", image_uri="img:1", model_data_url="s3://b/model.tar.gz",
                       instance_type="ml.g5.xlarge", env={"MDP_MAX_CONCURRENCY": 2})
    assert out["status"] == "InService"


def test_autoscaling_and_alarm_requests_are_valid(cfg):
    d = _deployer(cfg)
    with Stubber(d.aas) as aas, Stubber(d.cw) as cw:
        aas.add_response("register_scalable_target", {}, None)
        aas.add_response("put_scaling_policy", {"PolicyARN": "arn:p", "Alarms": []}, None)
        cw.add_response("put_metric_alarm", {}, None)
        cw.add_response("put_metric_alarm", {}, None)
        s = d.configure_autoscaling("mdp-x", min_capacity=1, max_capacity=4, target_invocations=50,
                                    scale_in_cooldown=600, scale_out_cooldown=120)
        alarms = d.create_alarms("mdp-x", p95_latency_ms=2000, error_rate_pct=5, sns_topic_arn=None)
    assert s["resource_id"] == "endpoint/mdp-x/variant/AllTraffic"
    assert alarms == ["mdp-x-p95-latency", "mdp-x-5xx-rate"]
