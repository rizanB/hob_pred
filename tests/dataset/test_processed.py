import numpy as np
import pandas as pd
import joblib
import pytest
import os

processed_data_path = "/home/rizanb/Documents/hob_pred/data/processed/"
processed_files = ["X_train_scaled.joblib", "X_test_scaled.joblib", "y_train.joblib", "y_test.joblib"]

# load data
@pytest.fixture(scope="module")
def data_files():
    data = {}
    
    data["X_train"] = joblib.load(os.path.join(processed_data_path, "X_train_scaled.joblib"))
    data["X_test"] = joblib.load(os.path.join(processed_data_path, "X_test_scaled.joblib"))
    data["y_train"] = joblib.load(os.path.join(processed_data_path, "y_train.joblib"))
    data["y_test"] = joblib.load(os.path.join(processed_data_path, "y_test.joblib"))

    return data
    
class TestDatasetProperties:
    
    def test_data_file_dir_exists(self):
        for f in processed_files:
            assert os.path.exists(os.path.join(processed_data_path, f)), f"processing files for trainig and testing missing!"
            
    def test_data_types(self, data_files):
        assert isinstance(data_files["X_train"], np.ndarray), f"X_train: {type(data_files["X_train"])} is not numpy array"
        assert isinstance(data_files["X_test"], np.ndarray), f"X_test: {type(data_files["X_test"])} is not numpy array"
        assert isinstance(data_files["y_train"], pd.Series ), f"y_train: {type(data_files["y_train"])} is not Pandas series"
        assert isinstance(data_files["y_test"], pd.Series), f"y_test: {type(data_files["y_test"])} is not Pandas series"
        
    def test_xy_shapes(self, data_files):
        len_X_train = data_files["X_train"].shape[0]
        len_y_train = len(data_files["y_train"])
        
        assert len_X_train == len_y_train, f"number of rows in X_train: {len_X_train} and y_train: {len_y_train} don't match"
        
        len_X_test = data_files["X_test"].shape[0]
        len_y_test = len(data_files["y_test"])
        
        assert len_X_test == len_y_test, f"number of rows in X_test: {len_X_test} and y_test: {len_y_test} don't match"
        