# Foundations of Neural Networks: From Tensors to Computational Graphs

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?logo=tensorflow)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive, production‑ready collection of notebooks covering the mathematical and computational foundations of neural networks – from tensors and operations to computational graphs, automatic differentiation, and logistic regression. Designed to showcase technical depth, reproducibility, and scientific rigor for PhD applications in computational biology, AI, and data science.**

---

## Overview

This repository contains a curated set of **twelve polished Jupyter notebooks** that systematically cover the foundational concepts of neural networks. Each notebook follows a consistent, professional structure:

- **Clear narrative** – introduction, methodology, implementation, evaluation, and critical reflection.
- **Reproducibility** – fixed random seeds, no `!pip install` cells, consistent data loading.
- **Interactive visualisations** – using Matplotlib for plots, decision boundaries, and activation functions.
- **Modular code** – helper functions, docstrings, and PEP8 compliance.
- **Publication‑quality outputs** – static plots saved for portfolio inclusion.

All notebooks are designed to run in a Conda environment (`ai_env`) with Python 3.10. Dependencies include PyTorch, TensorFlow, scikit‑learn, and standard data science libraries – ensuring a clean, professional experience.

---

## Repository Structure

```text
11_FoundationsNeuralNetworks/
├── notebooks/
│   ├── 01_Tensors_Introduction.ipynb
│   ├── 02_TensorOperations.ipynb
│   ├── 03_ComputationalGraphs_Theory.ipynb
│   ├── 04_BuildingComputationalGraphs.ipynb
│   ├── 05_StructuredDataAnalysis_PCA.ipynb
│   ├── 06_CategoricalData.ipynb
│   ├── 07_UnstructuredData_Images_Text_Graphs.ipynb
│   ├── 08_DataLoader_MiniBatching.ipynb
│   ├── 09_Datasets_Overview.ipynb
│   ├── 10_ForwardMode_AutomaticDifferentiation.ipynb
│   ├── 11_ReverseMode_Backpropagation.ipynb
│   ├── 12_LogisticRegression_ComputationalGraph.ipynb
│   └── data/                             # Downloaded datasets (ignored by Git)
├── README.md                               # This file
```

## Notebooks Overview

### 01 – Tensors: Introduction
**Topic:** The fundamental data structure in deep learning.  
**Content:** Tensor ranks (scalar, vector, matrix, higher‑order), shapes, creation with NumPy, PyTorch, and TensorFlow, accessing elements, reshaping, data types, gradients, GPU support.  
**Key Techniques:** Tensor creation, reshaping, `requires_grad`.  
**Visualisation:** 3D vector plots.  
**Applications:** Foundation for all subsequent notebooks.

---

### 02 – Tensor Operations
**Topic:** Essential operations on tensors.  
**Content:** Transposition, permutation, addition, scalar multiplication, dot products, matrix multiplication, higher‑rank tensor multiplication, Hadamard product, outer product.  
**Key Techniques:** `torch.matmul()`, `torch.outer()`, `torch.transpose()`.  
**Visualisation:** N/A.  
**Applications:** Building blocks for neural network computations.

---

### 03 – Computational Graphs: Theory
**Topic:** Static vs dynamic computational graphs.  
**Content:** TensorFlow 1.x static graphs vs PyTorch dynamic graphs, forward pass, backward pass (gradients), session management.  
**Key Techniques:** `tf.placeholder`, `tf.Session`, `torch.autograd`.  
**Visualisation:** N/A.  
**Applications:** Understanding how deep learning frameworks compute gradients.

---

### 04 – Structured Data Analysis and PCA
**Topic:** Analysing structured data with Pandas and dimensionality reduction.  
**Content:** Loading CSV/SQL data, descriptive statistics, visualisation (bar charts, histograms, box plots, pie charts), correlation analysis, PCA, linear autoencoder.  
**Key Techniques:** `pandas`, `sklearn.decomposition.PCA`, `torch` autoencoder.  
**Visualisation:** Scatter plots, heatmaps, density plots.  
**Applications:** Exploratory data analysis, feature reduction.

---

### 05 – Unstructured Data: Images, Text, Graphs
**Topic:** Handling images, text, and graphs.  
**Content:** Image representation (grayscale, RGB, RGBA), flattening, tokenisation, vocabulary creation, word embeddings, training embeddings, graph adjacency matrices.  
**Key Techniques:** `torch.nn.Embedding`, tokenisation, `networkx`.  
**Visualisation:** Image grids, embedding scatter plots, graph visualisation.  
**Applications:** Preparing images, text, and graph data for deep learning.

---

### 06 – DataLoader and Mini‑Batching
**Topic:** Efficient data loading with mini‑batches.  
**Content:** Custom `Dataset` class, `DataLoader`, mini‑batch training, comparison of batch sizes (64 vs 1).  
**Key Techniques:** `torch.utils.data.Dataset`, `DataLoader`.  
**Visualisation:** N/A.  
**Applications:** Efficient neural network training.

---

### 07 – Automatic Differentiation: Forward and Reverse Modes
**Topic:** Forward‑mode and reverse‑mode automatic differentiation.  
**Content:** Chain rule implementation, custom `Node`, `Cos`, `Pow`, `Sum` classes, derivative propagation in both directions, gradient accumulation.  
**Key Techniques:** Forward propagation of derivatives, backward propagation (backpropagation).  
**Visualisation:** N/A.  
**Applications:** Understanding AD for neural network training.

---

### 08 – Logistic Regression with Computational Graphs
**Topic:** Building a logistic regression model using a custom computational graph.  
**Content:** `Linear`, `Tanh`, `Softmax`, `CrossEntropy` nodes, training loop, gradient descent, comparison of `Softmax(Linear)` vs `Softmax(Tanh(Linear))` on Iris.  
**Key Techniques:** Reverse‑mode AD, multiclass classification.  
**Visualisation:** Loss curves, classification reports.  
**Applications:** End‑to‑end implementation of a trainable model.

---

## Key Learnings and Insights

- **Tensors** – Fundamental data structure; understanding ranks, shapes, and operations is essential for deep learning.
- **Computational Graphs** – Static graphs (TF1) vs dynamic graphs (PyTorch); building graphs from scratch demystifies autograd.
- **Automatic Differentiation** – Forward mode is efficient for few inputs; reverse mode (backpropagation) is efficient for few outputs and is the backbone of neural network training.
- **Data Handling** – Structured data requires encoding and scaling; unstructured data (images, text) requires specialised preprocessing.
- **Data Loading** – `Dataset` and `DataLoader` enable efficient mini‑batch training; batch size affects training speed and convergence.
- **Logistic Regression** – A simple linear model outperforms a non‑linear one on small, linearly separable datasets; non‑linearities are not always beneficial.
- **Transferability** – The computational graph framework can be extended to deeper architectures and more complex models.

---

## Business & Research Applications

- Building custom neural network architectures.
- Understanding and debugging gradient flow.
- Efficient data handling and preprocessing.
- Implementing models from scratch for research or education.
- Benchmarking model performance on standard datasets (Iris, MNIST, CIFAR‑10).
- Developing interpretable machine learning systems.
- Teaching and learning the fundamentals of deep learning.