import pandas as pd
import re

def preprocess_configs(filepath):
    """Reads and cleans network configuration files."""
    with open(filepath, 'r') as f:
        config_text = f.read()

    # Basic cleaning (adjust as needed for specific config formats)
    config_text = re.sub(r'#.*', '', config_text)  # Remove comments
    config_text = re.sub(r'\s+', ' ', config_text).strip() # Remove extra spaces and newlines.
    return config_text

def preprocess_logs(filepath):
    """Reads and cleans network log files."""
    df = pd.read_csv(filepath)
    # Basic cleaning (adjust based on log format)
    # Example: convert timestamp to datetime, handle missing values, etc.
    return df
