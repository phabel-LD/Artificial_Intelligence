#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Utility Functions for AI/ML Data Preprocessing Notebooks
----------------------------------------------------------------
This module provides common helper functions that can be imported
and used across multiple notebooks.

Typical usage in a notebook:
    import sys
    sys.path.append('../scripts')
    from utils import set_seed, get_data_dir

Functions:
    - load_data_if_exists: Safely load a CSV file if it exists.
    - ensure_dir: Create a directory if it doesn't exist.
    - set_seed: Set random seeds for reproducibility.
    - get_data_dir: Return the path to the data directory.
    - get_notebooks_dir: Return the path to the notebooks directory.
    - print_section: Print a formatted section header.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd


def load_data_if_exists(filepath):
    """
    Load a CSV file into a pandas DataFrame if the file exists.

    Args:
        filepath (str or Path): Path to the CSV file.

    Returns:
        pandas.DataFrame or None: DataFrame if file exists, else None.
    """
    path = Path(filepath)
    if path.exists():
        return pd.read_csv(path)
    return None


def ensure_dir(path):
    """
    Create a directory and any missing parent directories.

    Args:
        path (str or Path): Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    """
    Set random seeds for NumPy and Python's random module.

    This helps ensure reproducibility of results across runs.

    Args:
        seed (int): The seed value to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    # If using PyTorch or TensorFlow, additional seed settings can be added here.


def get_data_dir():
    """
    Return the path to the data directory (project_root/data/).

    Returns:
        Path: The data directory path.
    """
    return Path(__file__).parent.parent / "data"


def get_notebooks_dir():
    """
    Return the path to the notebooks directory (project_root/notebooks/).

    Returns:
        Path: The notebooks directory path.
    """
    return Path(__file__).parent.parent / "notebooks"


def print_section(title, char="="):
    """
    Print a formatted section header for clearer console output.

    Args:
        title (str): The section title.
        char (str): The character to use for the border (default '=').
    """
    print("\n" + char * 60)
    print(f"{title:^60}")
    print(char * 60 + "\n")