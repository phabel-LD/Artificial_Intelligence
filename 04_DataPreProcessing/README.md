# Data Preprocessing Module

A comprehensive collection of Jupyter notebooks demonstrating essential data preprocessing and feature engineering techniques for AI/ML workflows. This module covers data cleaning, transformation, dimensionality reduction, clustering, text processing, and more.

## Notebooks Overview

| Notebook | Description | Key Libraries |
|----------|-------------|---------------|
| AudioProcessing.ipynb | Audio signal processing, feature extraction (MFCCs), and visualisation | librosa, matplotlib |
| InputDataCharacteristics.ipynb | Exploratory data analysis: distributions, outliers, missing values, categorical encoding | pandas, seaborn, sklearn |
| IntroDataProcessing.ipynb | End-to-end data pipeline: loading, cleaning, imputation, encoding, scaling | pandas, sklearn |
| Kmeans_WordCloud.ipynb | Text clustering with K-Means and word cloud visualisation per cluster | sklearn, wordcloud, nltk |
| PCA_KMEANS.ipynb | PCA for dimensionality reduction with K-Means clustering and biplot visualisation | sklearn, matplotlib |
| NLP.ipynb | NLP basics: regex extraction, lemmatisation, stemming, and vectorisation | nltk, spacy, sklearn |
| PCAscratch.ipynb | PCA implementation from scratch using NumPy (covariance matrix, eigendecomposition) | numpy, pandas |
| ImageProcessing.ipynb | Image processing fundamentals: thresholding, morphology, segmentation, affine transforms | OpenCV, matplotlib |
| DatasetGenerator.ipynb | Synthetic dataset generation using scikit-learn utilities | sklearn.datasets |
| DiabetesDatasetPrepocessing.ipynb | Preprocessing the Pima Indians Diabetes dataset: missing values, scaling, feature selection | pandas, sklearn |
| FeelingsAnalysis_alexa_reviews.ipynb | Sentiment analysis on Alexa reviews with text preprocessing and classification | nltk, sklearn |

## Project Structure
```text
DataPreProcessing/
├── notebooks/
│   ├── AudioProcessing.ipynb
│   ├── DatasetGenerator.ipynb
│   ├── DiabetesDatasetPrepocessing.ipynb
│   ├── FeelingsAnalysis_alexa_reviews.ipynb
│   ├── ImageProcessing.ipynb
│   ├── InputDataCharacteristics.ipynb
│   ├── IntroDataProcessing.ipynb
│   ├── Kmeans_WordCloud.ipynb
│   ├── NLP.ipynb
│   ├── PCA_KMEANS.ipynb
│   └── PCAscratch.ipynb
├── scripts/
│   ├── check_env.py
│   ├── download_datasets.py
│   └── utils.py
├── data/
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the repository

git clone https://github.com/phabel-LD/Artificial_Intelligence
cd Artificial_Intelligence/DataPreProcessing

### 2. Create and activate the Conda environment

conda create -n ai_env python=3.10
conda activate ai_env

### 3. Install dependencies

pip install -r requirements.txt
python -m spacy download es_core_news_sm

### 4. Download external datasets (optional)

python scripts/download_datasets.py

### 5. Launch Jupyter Notebook

jupyter notebook

## Scripts

| Script | Purpose |
|--------|---------|
| check_env.py | Verify that all required packages are installed and importable |
| download_datasets.py | Download external datasets (currently supports Kaggle) |
| utils.py | Shared helper functions for notebooks (seed setting, file handling, etc.) |

## Dependencies

- Python 3.10+
- See requirements.txt for the full dependency list

## Usage

Each notebook is self-contained and can be run independently. Some notebooks use synthetic data; others load external datasets (automatically downloaded via download_datasets.py).

## License

MIT License – see the LICENSE file for details.

## Author

**Phabel Antonio Lopez-Delgado, BSc.**

- GitHub: phabel-LD
- Email: phabel@lcg.unam.mx

## Acknowledgments

- The kagglehub library for simplifying Kaggle dataset downloads
- The open-source libraries that make data science accessible to everyone