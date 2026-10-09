"""Approximate on-demand prices (USD/hour, us-east-1, Linux).

Used only for plan cost estimates and the max_hourly_usd guardrail. Verify
current prices at https://aws.amazon.com/ec2/pricing/ and
https://aws.amazon.com/sagemaker/pricing/ - edit this table if they drift.
"""

EC2_HOURLY = {
    "t3.xlarge": 0.17,
    "m6i.2xlarge": 0.38,
    "c6i.4xlarge": 0.68,
    "g4dn.xlarge": 0.53,
    "g4dn.2xlarge": 0.75,
    "g5.xlarge": 1.01,
    "g5.2xlarge": 1.21,
    "g5.4xlarge": 1.62,
    "g5.12xlarge": 5.67,
    "g6.xlarge": 0.80,
    "g6.2xlarge": 0.98,
    "g6e.xlarge": 1.86,
    "p4d.24xlarge": 32.77,
}

SAGEMAKER_HOURLY = {
    "ml.m5.xlarge": 0.23,
    "ml.c5.2xlarge": 0.41,
    "ml.g4dn.xlarge": 0.74,
    "ml.g5.xlarge": 1.41,
    "ml.g5.2xlarge": 1.52,
    "ml.g5.4xlarge": 2.03,
    "ml.g5.12xlarge": 7.09,
    "ml.g6.xlarge": 1.01,
    "ml.g6.2xlarge": 1.22,
    "ml.p4d.24xlarge": 37.69,
}


def ec2_hourly(instance_type: str) -> float | None:
    return EC2_HOURLY.get(instance_type)


def sagemaker_hourly(instance_type: str) -> float | None:
    return SAGEMAKER_HOURLY.get(instance_type)
