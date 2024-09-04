import os
import pandas as pd

def read_csv(file_path):
    """Read a CSV file and return a DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File {file_path} does not exist.")
        return pd.DataFrame()

def write_csv(df, file_path, mode='w'):
    """Write a DataFrame to a CSV file."""
    if not os.path.isfile(file_path) or mode == 'w':
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode=mode, header=False, index=False)
