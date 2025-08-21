"""
Configuration file for the MLOps Automated Pipeline.

This file contains all the configurable parameters for the SageMaker pipeline,
including instance types, hyperparameters, and thresholds.
"""

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)

# Pipeline Configuration
class PipelineConfig:
    """Configuration class for the MLOps pipeline."""
    
    # Basic Pipeline Settings
    BASE_JOB_PREFIX = "ab3-autompg-example"
    PIPELINE_NAME = "AutoMPGPipeline"
    MODEL_PACKAGE_GROUP_NAME = "AutoMpgPackageGroupName"
    
    # Instance Configuration
    DEFAULT_PROCESSING_INSTANCE_COUNT = 1
    DEFAULT_TRAINING_INSTANCE_TYPE = "ml.m5.xlarge"
    DEFAULT_PROCESSING_INSTANCE_TYPE = "ml.m5.xlarge"
    DEFAULT_INFERENCE_INSTANCE_TYPE = "ml.t2.medium"
    DEFAULT_TRANSFORM_INSTANCE_TYPE = "ml.m5.large"
    
    # Model Configuration
    DEFAULT_MODEL_APPROVAL_STATUS = "PendingManualApproval"
    DEFAULT_MSE_THRESHOLD = 46.0
    
    # XGBoost Hyperparameters (Fixed)
    XGBOOST_HYPERPARAMETERS = {
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
        "num_round": 50,
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weight": 6,
        "subsample": 0.7,
        "silent": 0,
    }
    
    # Hyperparameter Tuning Configuration
    TUNING_CONFIG = {
        "objective_metric_name": "validation:rmse",
        "objective_type": "Minimize",
        "max_jobs": 3,
        "max_parallel_jobs": 3,
        "strategy": "Bayesian",
    }
    
    # Hyperparameter Ranges for Tuning
    HYPERPARAMETER_RANGES = {
        "alpha": {"min": 0.01, "max": 10, "scaling_type": "Logarithmic"},
        "lambda": {"min": 0.01, "max": 10, "scaling_type": "Logarithmic"},
    }
    
    # Framework Versions
    SAGEMAKER_VERSION = "2.224.0"
    SKLEARN_VERSION = "1.2-1"
    XGBOOST_VERSION = "1.0-1"
    PYTHON_VERSION = "py3"
    
    # Data Configuration
    DATA_CONFIG = {
        "train_split": 0.7,
        "validation_split": 0.15,
        "test_split": 0.15,
        "content_type": "text/csv",
    }
    
    # Feature Configuration for Auto MPG Dataset
    FEATURE_COLUMNS = [
        "mpg", "cylinders", "displacement", "horsepower", 
        "weight", "acceleration", "model_year", "origin"
    ]
    LABEL_COLUMN = "mpg"
    
    # Cache Configuration
    CACHE_CONFIG = {
        "enable_caching": True,
        "expire_after": "30d"
    }

def get_pipeline_parameters(input_data_uri: str):
    """
    Get SageMaker pipeline parameters.
    
    Args:
        input_data_uri: S3 URI for the input dataset
        
    Returns:
        Dictionary of pipeline parameters
    """
    return {
        "processing_instance_count": ParameterInteger(
            name="ProcessingInstanceCount", 
            default_value=PipelineConfig.DEFAULT_PROCESSING_INSTANCE_COUNT
        ),
        "instance_type": ParameterString(
            name="TrainingInstanceType", 
            default_value=PipelineConfig.DEFAULT_TRAINING_INSTANCE_TYPE
        ),
        "model_approval_status": ParameterString(
            name="ModelApprovalStatus", 
            default_value=PipelineConfig.DEFAULT_MODEL_APPROVAL_STATUS
        ),
        "input_data": ParameterString(
            name="InputData", 
            default_value=input_data_uri
        ),
        "mse_threshold": ParameterFloat(
            name="MseThreshold", 
            default_value=PipelineConfig.DEFAULT_MSE_THRESHOLD
        ),
    }

def get_supported_instance_types():
    """Get list of supported instance types for different components."""
    return {
        "processing": ["ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge"],
        "training": ["ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge", "ml.c5.xlarge"],
        "inference": ["ml.t2.medium", "ml.m5.large", "ml.m5.xlarge"],
        "transform": ["ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge"],
    }
