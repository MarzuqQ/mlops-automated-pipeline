"""
Configuration package for the MLOps Automated Pipeline.

This package contains configuration files and parameters for the pipeline.
"""

from .pipeline_config import PipelineConfig, get_pipeline_parameters, get_supported_instance_types

__all__ = ["PipelineConfig", "get_pipeline_parameters", "get_supported_instance_types"]
