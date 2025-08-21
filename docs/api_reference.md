# API Reference

## Pipeline Configuration

### PipelineConfig Class

The main configuration class that contains all pipeline settings and parameters.

```python
from config.pipeline_config import PipelineConfig

# Access configuration values
config = PipelineConfig()
print(config.BASE_JOB_PREFIX)  # "ab3-autompg-example"
print(config.PIPELINE_NAME)    # "AutoMPGPipeline"
```

#### Class Attributes

| Attribute | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `BASE_JOB_PREFIX` | str | "ab3-autompg-example" | Base prefix for all SageMaker jobs |
| `PIPELINE_NAME` | str | "AutoMPGPipeline" | Name of the SageMaker pipeline |
| `MODEL_PACKAGE_GROUP_NAME` | str | "AutoMpgPackageGroupName" | Model package group name |
| `DEFAULT_PROCESSING_INSTANCE_COUNT` | int | 1 | Number of processing instances |
| `DEFAULT_TRAINING_INSTANCE_TYPE` | str | "ml.m5.xlarge" | Instance type for training |
| `DEFAULT_MSE_THRESHOLD` | float | 46.0 | MSE threshold for model registration |

#### XGBoost Hyperparameters

```python
# Access XGBoost hyperparameters
hyperparams = PipelineConfig.XGBOOST_HYPERPARAMETERS
print(hyperparams["eval_metric"])  # "rmse"
print(hyperparams["num_round"])    # 50
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `eval_metric` | "rmse" | Evaluation metric |
| `objective` | "reg:squarederror" | Training objective |
| `num_round` | 50 | Number of boosting rounds |
| `max_depth` | 5 | Maximum tree depth |
| `eta` | 0.2 | Learning rate |
| `gamma` | 4 | Minimum loss reduction |
| `min_child_weight` | 6 | Minimum sum of instance weight |
| `subsample` | 0.7 | Subsample ratio |

## Pipeline Functions

### get_pipeline_parameters()

Creates SageMaker pipeline parameters with proper typing.

```python
from config.pipeline_config import get_pipeline_parameters

# Create pipeline parameters
params = get_pipeline_parameters("s3://my-bucket/data.csv")

# Access individual parameters
print(params["processing_instance_count"].default_value)  # 1
print(params["instance_type"].default_value)             # "ml.m5.xlarge"
```

**Parameters:**
- `input_data_uri` (str): S3 URI for the input dataset

**Returns:**
- `dict`: Dictionary of SageMaker pipeline parameters

### get_supported_instance_types()

Returns supported instance types for different pipeline components.

```python
from config.pipeline_config import get_supported_instance_types

# Get supported instance types
instance_types = get_supported_instance_types()

print(instance_types["processing"])  # ["ml.m5.xlarge", "ml.m5.2xlarge", ...]
print(instance_types["training"])    # ["ml.m5.xlarge", "ml.m5.2xlarge", ...]
```

**Returns:**
- `dict`: Dictionary with instance types for each component

## Data Processing Scripts

### preprocess.py

Data preprocessing script for the Auto MPG dataset.

#### Command Line Interface

```bash
python src/preprocess.py --input-data s3://bucket/data.csv
```

**Arguments:**
- `--input-data` (required): S3 URI of the input dataset

#### Key Functions

##### merge_two_dicts(x, y)

Utility function to merge two dictionaries.

```python
from src.preprocess import merge_two_dicts

dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
result = merge_two_dicts(dict1, dict2)
# Returns: {"a": 1, "b": 2, "c": 3, "d": 4}
```

#### Data Schema

##### Feature Columns

| Column | Type | Description |
|--------|------|-------------|
| `mpg` | float64 | Miles per gallon (target variable) |
| `cylinders` | float64 | Number of cylinders |
| `displacement` | float64 | Engine displacement |
| `horsepower` | float64 | Engine horsepower |
| `weight` | float64 | Vehicle weight |
| `acceleration` | float64 | Acceleration time |
| `model_year` | float64 | Model year |
| `origin` | str | Country of origin |

##### Output Files

- `train/train.csv`: Training dataset (70% of data)
- `validation/validation.csv`: Validation dataset (15% of data)
- `test_set_1/test_set_1.csv`: Test dataset with labels (7.5% of data)
- `test_set_2/test_set_2.csv`: Test dataset without labels (7.5% of data)

### evaluate.py

Model evaluation script for calculating performance metrics.

#### Expected File Structure

The script expects the following file structure in the processing container:

```
/opt/ml/processing/
├── model/
│   └── model.tar.gz          # Compressed model artifact
├── test_set_1/
│   └── test_set_1.csv        # Test data with labels
└── evaluation/               # Output directory
    └── evaluation.json       # Generated metrics
```

#### Output Format

The evaluation script generates a JSON file with the following structure:

```json
{
  "regression_metrics": {
    "mse": {
      "value": 0.063,
      "standard_deviation": 0.236
    }
  }
}
```

## SageMaker Pipeline Components

### Pipeline Steps

#### ProcessingStep: PreprocessAutoMpgData

Data preprocessing step using SKLearnProcessor.

```python
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep

# Create processor
processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=role
)

# Create processing step
step = ProcessingStep(
    name="PreprocessAutoMpgData",
    step_args=processor_args
)
```

#### TuningStep: HPTuning

Hyperparameter tuning step using XGBoost.

```python
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.steps import TuningStep

# Create tuner
tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:rmse",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=3,
    max_parallel_jobs=3
)

# Create tuning step
step = TuningStep(
    name="HPTuning",
    step_args=tuner.fit(inputs)
)
```

#### ProcessingStep: EvaluateTopModel

Model evaluation step using ScriptProcessor.

```python
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

# Create processor
processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type="ml.m5.xlarge",
    role=role
)

# Create evaluation step
step = ProcessingStep(
    name="EvaluateTopModel",
    step_args=processor_args,
    property_files=[evaluation_report]
)
```

#### ConditionStep: AutoMPGMSECond

Conditional step for model registration based on performance.

```python
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep

# Create condition
condition = ConditionLessThanOrEqualTo(
    left=JsonGet(
        step_name="EvaluateTopModel",
        property_file=evaluation_report,
        json_path="regression_metrics.mse.value"
    ),
    right=mse_threshold
)

# Create condition step
step = ConditionStep(
    name="AutoMPGMSECond",
    conditions=[condition],
    if_steps=[register_step, transform_step],
    else_steps=[fail_step]
)
```

### Pipeline Parameters

#### Parameter Types

```python
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)

# Integer parameter
processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1
)

# String parameter
instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.xlarge"
)

# Float parameter
mse_threshold = ParameterFloat(
    name="MseThreshold",
    default_value=46.0
)
```

## Error Handling

### Common Errors

#### Data Processing Errors

```python
# Handle missing input data
try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    logger.error("Input data file not found: %s", input_path)
    raise

# Handle data type conversion errors
try:
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
except Exception as e:
    logger.error("Error converting horsepower column: %s", e)
    raise
```

#### Model Loading Errors

```python
# Handle model loading errors
try:
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = pickle.load(open("xgboost-model", "rb"))
except (tarfile.TarError, FileNotFoundError, pickle.PickleError) as e:
    logger.error("Error loading model: %s", e)
    raise
```

### Logging Configuration

```python
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Log messages
logger.info("Starting preprocessing")
logger.debug("Processing %d rows", len(df))
logger.error("Failed to process data: %s", error_message)
```

## Best Practices

### Configuration Management

1. Use the centralized `PipelineConfig` class
2. Override defaults through environment variables
3. Validate configuration parameters before use

### Error Handling

1. Use try-catch blocks for external operations
2. Log errors with appropriate severity levels
3. Fail fast for critical errors

### Resource Management

1. Choose appropriate instance types for workload
2. Use spot instances for cost optimization
3. Enable caching for frequently used steps

### Security

1. Use IAM roles with minimal permissions
2. Encrypt data at rest and in transit
3. Regularly rotate access keys

### Monitoring

1. Enable CloudWatch logging for all components
2. Set up alerts for pipeline failures
3. Monitor resource utilization and costs
