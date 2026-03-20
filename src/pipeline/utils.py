import os
import joblib
import pandas as pd

def save_pickle(obj, path):
    """Save a Python object to a pickle file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path):
    """Load a pickled Python object."""
    return joblib.load(path)

def load_csv(path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)

def save_csv(df, path, index=False):
    """Save a DataFrame to CSV, ensuring the directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
