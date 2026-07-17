#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Downloader for AI/ML Data Preprocessing Notebooks
--------------------------------------------------------
This script downloads external datasets used by the notebooks.
Currently, it supports Kaggle datasets via the kagglehub library.

The datasets are stored in the 'data/' directory at the project root.
Each dataset is placed in a subdirectory named after the dataset's local name.

Usage:
    python scripts/download_datasets.py

The script will skip downloading if the target directory already exists.
"""

import shutil
import zipfile
from pathlib import Path

import kagglehub


def download_kaggle_datasets():
    """
    Download all Kaggle datasets defined in the DATASETS dictionary.

    The dictionary maps Kaggle dataset identifiers (as used by kagglehub)
    to a local folder name under the data/ directory.

    For each dataset:
      - If the local target already exists, skip download.
      - If the downloaded path is a zip file, extract it.
      - Otherwise, copy the entire directory.
    """
    # Define datasets: {kaggle_id: local_folder_name}
    datasets = {
        "efrodl/german-rap-dataset": "german_rap_lyrics",
    }

    # Ensure the data directory exists
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    for dataset, local_name in datasets.items():
        print(f"Processing {dataset}...")
        target = data_dir / local_name

        # Skip if already downloaded
        if target.exists():
            print(f"  Already exists: {target}")
            continue

        try:
            # Download using kagglehub
            downloaded_path = kagglehub.dataset_download(dataset)
            print(f"  Downloaded to: {downloaded_path}")

            # Determine if it's a zip or a directory
            if downloaded_path.endswith(".zip"):
                # Extract zip file
                with zipfile.ZipFile(downloaded_path, "r") as zip_ref:
                    zip_ref.extractall(target)
                print(f"  Extracted to: {target}")
            else:
                # Copy directory
                shutil.copytree(downloaded_path, target)
                print(f"  Copied to: {target}")
        except Exception as e:
            print(f"  Failed to download {dataset}: {e}")


def main():
    """
    Main entry point: download all datasets.
    """
    print("=" * 50)
    print("DATASET DOWNLOADER")
    print("=" * 50)
    download_kaggle_datasets()
    print("Done.")


if __name__ == "__main__":
    main()