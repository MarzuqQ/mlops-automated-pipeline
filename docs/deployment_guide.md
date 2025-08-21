# Deployment Guide

This guide provides step-by-step instructions for deploying the MLOps Automated Pipeline in different environments.

## Prerequisites

### AWS Account Setup

1. **AWS Account**: Ensure you have an active AWS account with billing enabled
2. **IAM Permissions**: Your user/role needs the following permissions:
   - SageMaker full access
   - S3 read/write access
   - IAM role creation and attachment
   - CloudWatch logs access

3. **Service Limits**: Verify your account has sufficient service limits:
   - SageMaker processing instances: 5+
   - SageMaker training instances: 5+
   - S3 storage: 100GB+

### Local Development Environment

```bash
# Python 3.10 or higher
python --version  # Should be 3.10+

# AWS CLI (optional but recommended)
aws --version

# Git
git --version
```

## Environment Setup

### 1. Development Environment

For local development and testing:

```bash
# Clone the repository
git clone https://github.com/yourusername/mlops-automated-pipeline.git
cd mlops-automated-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sagemaker; print(sagemaker.__version__)"
```

### 2. AWS SageMaker Studio

For cloud-based development:

1. **Create SageMaker Domain**:
   ```bash
   aws sagemaker create-domain \
     --domain-name mlops-pipeline-domain \
     --auth-mode IAM \
     --default-user-settings ExecutionRole=arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole
   ```

2. **Create User Profile**:
   ```bash
   aws sagemaker create-user-profile \
     --domain-id d-xxxxxxxxxxxx \
     --user-profile-name mlops-user
   ```

3. **Launch Studio** and clone the repository in the terminal

### 3. Production Environment

For production deployments:

```bash
# Use AWS CloudFormation or Terraform
# Example CloudFormation template provided below
```

## IAM Role Configuration

### SageMaker Execution Role

Create an IAM role with the following trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Attach the following managed policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or more restrictive bucket-specific policy)

Custom policy for additional permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

## Data Setup

### 1. Prepare Your Dataset

The pipeline expects the Auto MPG dataset in CSV format. You can:

1. **Use the original dataset**:
   ```bash
   # Download from UCI repository
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
   
   # Convert to CSV format (add headers if needed)
   # Upload to S3
   aws s3 cp auto-mpg.csv s3://your-bucket/auto-mpg.csv
   ```

2. **Use your own dataset**: Ensure it follows the expected schema (see API Reference)

### 2. S3 Bucket Setup

Create and configure your S3 bucket:

```bash
# Create bucket
aws s3 mb s3://your-mlops-pipeline-bucket

# Set up bucket policy (optional - for cross-account access)
aws s3api put-bucket-policy \
  --bucket your-mlops-pipeline-bucket \
  --policy file://bucket-policy.json

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
  --bucket your-mlops-pipeline-bucket \
  --versioning-configuration Status=Enabled
```

## Pipeline Deployment

### 1. Configuration

Update the configuration file:

```python
# config/pipeline_config.py
class PipelineConfig:
    # Update with your specific values
    BASE_JOB_PREFIX = "your-company-autompg"
    PIPELINE_NAME = "YourAutoMPGPipeline"
    
    # Adjust instance types based on your needs
    DEFAULT_TRAINING_INSTANCE_TYPE = "ml.m5.xlarge"  # or larger
    DEFAULT_MSE_THRESHOLD = 46.0  # adjust based on requirements
```

### 2. Deploy Pipeline

#### Option A: Using Jupyter Notebook

1. Open `automated-ml-pipeline.ipynb`
2. Update the configuration variables:
   ```python
   # Update these variables
   input_data_uri = "s3://your-bucket/auto-mpg.csv"
   default_bucket = "your-mlops-pipeline-bucket"
   role = "arn:aws:iam::ACCOUNT:role/YourSageMakerRole"
   ```
3. Run all cells to create and execute the pipeline

#### Option B: Using Python Script

Create a deployment script:

```python
# deploy_pipeline.py
import boto3
import sagemaker
from config.pipeline_config import PipelineConfig, get_pipeline_parameters

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Configure pipeline
input_data_uri = "s3://your-bucket/auto-mpg.csv"
pipeline_params = get_pipeline_parameters(input_data_uri)

# Create pipeline (implementation details in notebook)
pipeline = create_pipeline(pipeline_params, role, sagemaker_session)

# Deploy pipeline
pipeline.upsert(role_arn=role)
print(f"Pipeline {pipeline.name} created successfully!")

# Execute pipeline
execution = pipeline.start()
print(f"Pipeline execution started: {execution.arn}")
```

Run the deployment:

```bash
python deploy_pipeline.py
```

### 3. Monitor Deployment

```bash
# List pipelines
aws sagemaker list-pipelines

# Describe pipeline
aws sagemaker describe-pipeline --pipeline-name YourAutoMPGPipeline

# List executions
aws sagemaker list-pipeline-executions --pipeline-name YourAutoMPGPipeline

# Monitor specific execution
aws sagemaker describe-pipeline-execution --pipeline-execution-arn arn:aws:sagemaker:...
```

## Production Considerations

### 1. Security Hardening

#### Network Security
```bash
# Create VPC endpoint for SageMaker (optional)
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.region.sagemaker.api \
  --route-table-ids rtb-12345678
```

#### Encryption
- Enable S3 bucket encryption
- Use KMS keys for SageMaker jobs
- Enable CloudWatch log encryption

#### Access Control
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT:role/MLOpsTeamRole"
      },
      "Action": [
        "sagemaker:StartPipelineExecution",
        "sagemaker:DescribePipeline*",
        "sagemaker:ListPipeline*"
      ],
      "Resource": "arn:aws:sagemaker:*:*:pipeline/your-pipeline-name"
    }
  ]
}
```

### 2. Monitoring and Alerting

#### CloudWatch Alarms

```bash
# Pipeline failure alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "MLOps-Pipeline-Failures" \
  --alarm-description "Alert on pipeline failures" \
  --metric-name "PipelineExecutionFailure" \
  --namespace "AWS/SageMaker" \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --alarm-actions arn:aws:sns:region:account:alert-topic
```

#### Custom Metrics

```python
import boto3

# Publish custom metrics
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='MLOps/Pipeline',
    MetricData=[
        {
            'MetricName': 'ModelAccuracy',
            'Value': model_accuracy,
            'Unit': 'Percent'
        },
    ]
)
```

### 3. Cost Optimization

#### Spot Instances

```python
# Use spot instances for training
xgb_train = Estimator(
    # ... other parameters
    use_spot_instances=True,
    max_wait=3600,  # Wait up to 1 hour for spot
    max_run=1800,   # Max training time
)
```

#### Pipeline Caching

```python
from sagemaker.workflow.steps import CacheConfig

# Enable caching
cache_config = CacheConfig(
    enable_caching=True,
    expire_after="30d"
)

# Apply to steps
step_process = ProcessingStep(
    name="PreprocessAutoMpgData",
    step_args=processor_run_args,
    cache_config=cache_config
)
```

### 4. Backup and Disaster Recovery

#### Automated Backups

```bash
# S3 Cross-Region Replication
aws s3api put-bucket-replication \
  --bucket your-mlops-pipeline-bucket \
  --replication-configuration file://replication-config.json

# Pipeline Definition Backup
aws sagemaker describe-pipeline \
  --pipeline-name YourAutoMPGPipeline > pipeline-backup.json
```

#### Recovery Procedures

1. **Data Recovery**: Restore from S3 versioned backups
2. **Pipeline Recovery**: Recreate from saved definition
3. **Model Recovery**: Restore from Model Registry

## Troubleshooting

### Common Issues

#### 1. Permission Errors

```bash
# Error: Access Denied
# Solution: Check IAM role permissions

# Verify role
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://your-bucket/
```

#### 2. Resource Limits

```bash
# Error: ResourceLimitExceeded
# Solution: Request limit increase or use different instance types

# Check current limits
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-1194F20D  # Processing instances
```

#### 3. Data Format Issues

```python
# Error: Data parsing failed
# Solution: Validate data format

import pandas as pd

# Check data format
df = pd.read_csv('your-data.csv')
print(df.info())
print(df.head())

# Validate schema
expected_columns = ['mpg', 'cylinders', 'displacement', ...]
assert all(col in df.columns for col in expected_columns)
```

### Debugging Steps

1. **Check CloudWatch Logs**:
   ```bash
   aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker
   ```

2. **Examine Pipeline Execution**:
   ```python
   execution.list_steps()
   failed_steps = [step for step in execution.list_steps() if step['StepStatus'] == 'Failed']
   ```

3. **Validate Configuration**:
   ```python
   from config.pipeline_config import PipelineConfig
   
   # Validate configuration
   config = PipelineConfig()
   assert config.DEFAULT_MSE_THRESHOLD > 0
   assert config.DEFAULT_TRAINING_INSTANCE_TYPE.startswith('ml.')
   ```

## Scaling Considerations

### Horizontal Scaling

- Increase `max_parallel_jobs` for hyperparameter tuning
- Use multiple processing instances
- Implement data partitioning for large datasets

### Vertical Scaling

- Use larger instance types for memory-intensive operations
- Consider GPU instances for deep learning workloads
- Optimize batch sizes for processing

### Multi-Region Deployment

```bash
# Deploy in multiple regions for high availability
regions=("us-east-1" "us-west-2" "eu-west-1")

for region in "${regions[@]}"; do
    aws sagemaker create-pipeline \
      --region $region \
      --pipeline-name YourAutoMPGPipeline-$region \
      --pipeline-definition file://pipeline-definition.json
done
```

## Maintenance

### Regular Tasks

1. **Update Dependencies**: Monthly security updates
2. **Monitor Costs**: Weekly cost analysis
3. **Performance Review**: Monthly pipeline performance review
4. **Security Audit**: Quarterly security assessment

### Automated Maintenance

```python
# Automated pipeline cleanup
import boto3
from datetime import datetime, timedelta

sagemaker = boto3.client('sagemaker')

# Delete old executions (older than 30 days)
cutoff_date = datetime.now() - timedelta(days=30)

executions = sagemaker.list_pipeline_executions(
    PipelineName='YourAutoMPGPipeline'
)

for execution in executions['PipelineExecutionSummaries']:
    if execution['CreationTime'] < cutoff_date:
        print(f"Cleaning up execution: {execution['PipelineExecutionArn']}")
        # Implement cleanup logic
```

This deployment guide provides comprehensive instructions for setting up and maintaining the MLOps pipeline in various environments. Follow the appropriate sections based on your deployment scenario and requirements.
