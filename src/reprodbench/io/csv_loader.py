from pathlib import Path

import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load a dataset from a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): The path to the CSV file.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    return pd.read_csv(csv_path)
