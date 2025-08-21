import argparse
import logging
import os
import pathlib
import boto3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Define column names and data types for the Auto MPG dataset
feature_columns_names = [
    "mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"
]
label_column = "mpg"

feature_columns_dtype = {
    "cylinders": np.float64,  # Change to float64 to handle NA values before converting to int
    "displacement": np.float64,
    "horsepower": np.float64,
    "weight": np.float64,
    "acceleration": np.float64,
    "model_year": np.float64,  # Change to float64 to handle NA values before converting to int
    "origin": str,
}
label_column_dtype = {"mpg": np.float64}

def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)  # Create directories for input and output
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/auto-mpg-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)  # Download the dataset from S3

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn, header=None, names=feature_columns_names)  # Read the CSV file into a DataFrame
    os.unlink(fn)  # Delete the local file after reading

    # Ensure 'horsepower' is handled correctly
    logger.debug("Handling missing values in 'horsepower' column.")
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')  # Convert 'horsepower' to numeric, coerce errors to NaN

    # Convert non-numeric values to NaN
    for col in feature_columns_dtype.keys():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values before converting data types
    df = df.fillna(df.median())  # Fill NaN values with the median

    # Convert columns to the appropriate data types
    for col, dtype in feature_columns_dtype.items():
        df[col] = df[col].astype(dtype)

    logger.debug("Defining transformers.")
    features = list(feature_columns_names)
    features.remove(label_column)  # Remove the label column from the features list

    numeric_features = [name for name in features if df.dtypes[name] != 'object']  # Identify numeric features
    categorical_features = [name for name in features if df.dtypes[name] == 'object']  # Identify categorical features

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with median
            ("scaler", StandardScaler()),  # Standardize the numeric features
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # Impute missing values with 'missing'
            ("onehot", OneHotEncoder(handle_unknown="ignore")),  # One-hot encode categorical features
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),  # Apply numeric transformations
            ("cat", categorical_transformer, categorical_features),  # Apply categorical transformations
        ]
    )

    logger.info("Applying transforms.")
    y = df.pop(label_column)  # Separate the label from the features
    X_pre = preprocess.fit_transform(df)  # Fit and transform the features
    y_pre = y.to_numpy().reshape(len(y), 1)  # Reshape the label array

    X = np.concatenate((y_pre, X_pre), axis=1)  # Concatenate the label and features

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)  # Shuffle the dataset
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])  # Split the dataset

    # Further split the test set into test_set_1 and test_set_2
    test_set_1, test_set_2 = np.split(test, [int(0.5 * len(test))])

    logger.info("Writing out datasets to %s.", base_dir)
    pathlib.Path(f"{base_dir}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/validation").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test_set_1").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test_set_2").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)  # Save training set
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)  # Save validation set
    pd.DataFrame(test_set_1).to_csv(f"{base_dir}/test_set_1/test_set_1.csv", header=False, index=False)  # Save test_set_1

    # Remove the label column from test_set_2
    test_set_2_features = test_set_2[:, 1:]
    pd.DataFrame(test_set_2_features).to_csv(f"{base_dir}/test_set_2/test_set_2.csv", header=False, index=False)  # Save test_set_2 without the label column
