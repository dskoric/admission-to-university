import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path):
    """Load and preprocess the admission dataset.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    # Load data
    data = pd.read_csv(file_path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    
    # Separate features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    return X, y