import json
import pathlib
import tarfile
import pickle
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"  # Path to the tarred model file
    with tarfile.open(model_path) as tar:  # Extract the tarred model file
        tar.extractall(path=".")
    model = pickle.load(open("xgboost-model", "rb"))  # Load the XGBoost model from the extracted file

    test_path = "/opt/ml/processing/test_set_1/test_set_1.csv"  # Path to the test dataset
    df = pd.read_csv(test_path, header=None)  # Read the test dataset into a DataFrame
    y_test = df.iloc[:, 0].to_numpy()  # Extract the labels (assuming the label is the first column)
    df.drop(df.columns[0], axis=1, inplace=True)  # Drop the label column from the test data
    dmatrix_test = xgboost.DMatrix(df.values)  # Create DMatrix for XGBoost prediction

    predictions = model.predict(dmatrix_test)  # Make predictions using the loaded model
    mse = mean_squared_error(y_test, predictions)  # Calculate mean squared error
    std = np.std(y_test - predictions)  # Calculate standard deviation of errors

    report_dict = {  # Create a dictionary to store the evaluation metrics
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"                     # Define the output directory for the evaluation report
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)      # Create the output directory if it doesn't exist
    evaluation_path = f"{output_dir}/evaluation.json"                # Path to the evaluation report JSON file
    with open(evaluation_path, "w") as f:                            # Write the evaluation metrics to the JSON file
        f.write(json.dumps(report_dict))
