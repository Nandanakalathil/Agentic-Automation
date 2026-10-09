#!/usr/bin/env bash
# One-time AWS setup for the pipeline. Idempotent: safe to re-run.
# Creates: S3 artifact bucket, EC2 instance role/profile, SageMaker execution role, ECR repo.
# Usage: AWS_REGION=us-east-1 ./scripts/bootstrap_aws.sh [bucket-name]
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="${1:-model-pipeline-artifacts-${ACCOUNT}-${REGION}}"
EC2_ROLE=model-pipeline-ec2-role
SM_ROLE=model-pipeline-sagemaker-role
ECR_REPO=model-pipeline-endpoints

echo "Account $ACCOUNT, region $REGION, bucket $BUCKET"

# ---- S3 bucket (private, encrypted, artifacts expire after 30 days) --------
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{
  "Rules": [{"ID": "expire-ssm-and-workspace", "Status": "Enabled",
             "Filter": {"Prefix": "runs/"}, "Expiration": {"Days": 30}}]}'

# ---- EC2 instance role: SSM + bucket + ECR push -----------------------------
if ! aws iam get-role --role-name "$EC2_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$EC2_ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
fi
aws iam attach-role-policy --role-name "$EC2_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name "$EC2_ROLE" --policy-name model-pipeline-ec2 --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\", \"s3:ListBucket\"],
     \"Resource\": [\"arn:aws:s3:::${BUCKET}\", \"arn:aws:s3:::${BUCKET}/*\"]},
    {\"Effect\": \"Allow\", \"Action\": \"ecr:GetAuthorizationToken\", \"Resource\": \"*\"},
    {\"Effect\": \"Allow\", \"Action\": [\"ecr:BatchCheckLayerAvailability\", \"ecr:InitiateLayerUpload\",
       \"ecr:UploadLayerPart\", \"ecr:CompleteLayerUpload\", \"ecr:PutImage\", \"ecr:BatchGetImage\",
       \"ecr:GetDownloadUrlForLayer\"],
     \"Resource\": \"arn:aws:ecr:${REGION}:${ACCOUNT}:repository/${ECR_REPO}\"}
  ]}"
if ! aws iam get-instance-profile --instance-profile-name "$EC2_ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$EC2_ROLE"
  aws iam add-role-to-instance-profile --instance-profile-name "$EC2_ROLE" --role-name "$EC2_ROLE"
fi

# ---- SageMaker execution role: pull image, read model.tar.gz, write logs ---
if ! aws iam get-role --role-name "$SM_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$SM_ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
fi
aws iam put-role-policy --role-name "$SM_ROLE" --policy-name model-pipeline-sagemaker --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:ListBucket\"],
     \"Resource\": [\"arn:aws:s3:::${BUCKET}\", \"arn:aws:s3:::${BUCKET}/*\"]},
    {\"Effect\": \"Allow\", \"Action\": \"ecr:GetAuthorizationToken\", \"Resource\": \"*\"},
    {\"Effect\": \"Allow\", \"Action\": [\"ecr:BatchGetImage\", \"ecr:GetDownloadUrlForLayer\",
       \"ecr:BatchCheckLayerAvailability\"],
     \"Resource\": \"arn:aws:ecr:${REGION}:${ACCOUNT}:repository/${ECR_REPO}\"},
    {\"Effect\": \"Allow\", \"Action\": [\"logs:CreateLogGroup\", \"logs:CreateLogStream\", \"logs:PutLogEvents\",
       \"logs:DescribeLogStreams\"], \"Resource\": \"arn:aws:logs:${REGION}:${ACCOUNT}:log-group:/aws/sagemaker/*\"},
    {\"Effect\": \"Allow\", \"Action\": [\"cloudwatch:PutMetricData\"], \"Resource\": \"*\"}
  ]}"

# ---- ECR repository ----------------------------------------------------------
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null

echo
echo "Done. Add this to .env:"
echo "  PIPELINE_ARTIFACT_BUCKET=$BUCKET"
echo "  AWS_REGION=$REGION"
