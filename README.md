# MLOps Automated Pipeline

An end-to-end automated machine learning pipeline built with AWS SageMaker for predicting automobile fuel efficiency (MPG) using the Auto MPG dataset. This project demonstrates MLOps best practices including data preprocessing, hyperparameter tuning, model evaluation, conditional model registration, and batch transformation.

## 🚀 Features

- **Automated Data Processing**: Intelligent preprocessing with feature engineering and data splitting
- **Hyperparameter Optimization**: Bayesian optimization for XGBoost model tuning
- **Model Evaluation**: Automated model performance evaluation with MSE metrics
- **Conditional Model Registration**: Smart model registration based on performance thresholds
- **Batch Inference**: Automated batch transformation capabilities
- **Pipeline Caching**: Optimized execution with step caching for faster iterations
- **AWS Integration**: Full integration with AWS SageMaker services

## 📊 Architecture

The pipeline consists of the following components:

```
Data Input (S3) → Preprocessing → Hyperparameter Tuning → Model Evaluation → Conditional Logic → Model Registration/Batch Transform
```

### Pipeline Steps:

1. **Data Preprocessing** (`PreprocessAutoMpgData`)
   - Data cleaning and feature engineering
   - Train/validation/test split (70/15/15)
   - Feature scaling and encoding

2. **Hyperparameter Tuning** (`HPTuning`)
   - Bayesian optimization strategy
   - XGBoost model with RMSE optimization
   - Parallel job execution for efficiency

3. **Model Evaluation** (`EvaluateTopModel`)
   - Performance assessment on test set
   - MSE and standard deviation calculation
   - Automated report generation

4. **Conditional Model Registration** (`AutoMPGMSECond`)
   - MSE threshold-based decision making
   - Automatic model registration if performance meets criteria
   - Batch transformation setup for qualified models

## 🛠️ Technology Stack

- **AWS SageMaker**: ML pipeline orchestration
- **XGBoost**: Gradient boosting algorithm
- **scikit-learn**: Data preprocessing and evaluation
- **Python 3.10**: Core programming language
- **Boto3**: AWS SDK for Python
- **Pandas/NumPy**: Data manipulation and analysis

## 📋 Prerequisites

- AWS Account with SageMaker access
- IAM role with necessary SageMaker permissions
- Python 3.10+
- AWS CLI configured (optional)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mlops-automated-pipeline.git
cd mlops-automated-pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure AWS Credentials

Ensure your AWS credentials are configured either through:
- AWS CLI: `aws configure`
- Environment variables
- IAM roles (if running on AWS infrastructure)

### 4. Prepare Your Data

Upload your Auto MPG dataset to an S3 bucket:

```bash
aws s3 cp your-dataset.csv s3://your-bucket/auto-mpg.csv
```

### 5. Run the Pipeline

Execute the Jupyter notebook or run the pipeline programmatically:

```python
# Update the input_data parameter with your S3 URI
input_data_uri = "s3://your-bucket/auto-mpg.csv"

# Execute the pipeline
execution = pipeline.start()
execution.wait()
```

## 📁 Project Structure

```
mlops-automated-pipeline/
├── automated-ml-pipeline.ipynb    # Main pipeline notebook
├── src/
│   ├── preprocess.py             # Data preprocessing script
│   └── evaluate.py               # Model evaluation script
├── config/
│   └── pipeline_config.py        # Pipeline configuration
├── docs/
│   ├── architecture.md           # Architecture documentation
│   └── api_reference.md          # API reference
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT license
└── README.md                    # This file
```

## ⚙️ Configuration

### Pipeline Parameters

The pipeline accepts the following configurable parameters:

- `ProcessingInstanceCount`: Number of processing instances (default: 1)
- `TrainingInstanceType`: EC2 instance type for training (default: ml.m5.xlarge)
- `ModelApprovalStatus`: Model approval status (default: PendingManualApproval)
- `InputData`: S3 URI for input dataset
- `MseThreshold`: MSE threshold for model registration (default: 46.0)

### Hyperparameter Tuning

The pipeline tunes the following XGBoost hyperparameters:

- `alpha`: L1 regularization term (range: 0.01-10)
- `lambda`: L2 regularization term (range: 0.01-10)

Fixed hyperparameters:
- `eval_metric`: rmse
- `objective`: reg:squarederror
- `num_round`: 50
- `max_depth`: 5
- `eta`: 0.2

## 📈 Model Performance

The pipeline evaluates models using:

- **Primary Metric**: Root Mean Square Error (RMSE)
- **Secondary Metrics**: Mean Squared Error (MSE), Standard Deviation
- **Threshold-based Registration**: Models are automatically registered only if MSE ≤ configured threshold

## 🔄 Pipeline Execution

### Manual Execution

```python
# Start pipeline execution
execution = pipeline.start()

# Monitor execution
execution.describe()

# Wait for completion
execution.wait()

# List pipeline executions
pipeline.list_executions()
```

### Scheduled Execution

The pipeline can be integrated with:
- **AWS EventBridge**: For scheduled executions
- **AWS Lambda**: For event-driven triggers
- **SageMaker Projects**: For CI/CD integration

## 📊 Monitoring and Logging

- **SageMaker Studio**: Visual pipeline monitoring
- **CloudWatch Logs**: Detailed execution logs
- **SageMaker Experiments**: Experiment tracking
- **Model Registry**: Versioned model artifacts

This project is provided as-is for educational and reference purposes. Please refer to the comprehensive documentation included in the repository for implementation guidance.

---

**Built with ❤️ for the MLOps community**
