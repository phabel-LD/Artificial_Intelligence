# Multivariable Calculus Portfolio: Interactive Visualisations for AI & Optimisation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive collection of production‑ready Jupyter notebooks exploring multivariable calculus through interactive 3D visualisations. Designed to bridge foundational mathematics to machine learning, optimisation, and computational science – ideal for PhD applications in AI, computational biology, and data science.**

---

## Overview

This repository contains a **professional, well‑documented** series of notebooks that demonstrate mastery of multivariable calculus concepts through interactive visualisation, numerical computation, and connection to real‑world applications. Each notebook is self‑contained, modular, and follows rigorous coding standards (PEP8, docstrings, reproducibility seeds, and publication‑quality visualisations).

**Key Features:**
- **Interactive 3D visualisations** using `plotly` – toggle layers, rotate, zoom, and explore.
- **From‑scratch implementations** of core algorithms (gradient descent, Taylor series, line integrals).
- **Numerical integration** with `scipy` (dblquad, tplquad) for double and triple integrals.
- **Consistent structure** – introduction, helper functions, data generation, interactive plot, summary, next steps.
- **Fallback HTML export** – if interactive rendering fails, figures are saved as HTML files for browser viewing.

**Environment:** All notebooks run in a Conda environment (`ai_env`) with Python 3.10. Dependencies are pre‑installed – **no `!pip install` cells** – ensuring reproducibility and a clean professional appearance.

---

## Repository Structure

```text
02_MultivariableCalculus/
└── notebooks/
├── 01_SphereTangentPlanes.ipynb # Sphere geometry & tangent planes
├── 02_SaddlePointsAnalysis.ipynb # Critical points & Hessian classification
├── 03_DirectionalDerivatives.ipynb # Gradients & rates of change
├── 04_GradientDescend.ipynb # Optimisation algorithm visualisation
├── 05_LagrangeMultipliers.ipynb # Constrained optimisation
├── 06_TaylorSeries.ipynb # Local approximation in 2D
├── 07_VectorCalculus.ipynb # Grad, Div, Curl, Line/Double/Triple Integrals
└── plots/ # Saved figures (HTML & PNG)
```


---

## Notebooks Overview

### 01 – Sphere and Tangent Planes
**Topic:** Geometry of surfaces, gradients as normals.  
**Visualisation:** 3D sphere with up to six cardinal tangent planes. Interactive toggling of layers.  
**Key Techniques:** Parametric surfaces, tangent plane equation from gradient.  
**Applications:** Understanding local linear approximation, curvature, and surface normals.

---

### 02 – Saddle Point Analysis
**Topic:** Critical points, Hessian matrix, saddle classification.  
**Visualisation:** Hyperbolic paraboloid `z = x² - y²` with saddle plane and random plane.  
**Key Techniques:** Gradient, Hessian, critical point classification.  
**Applications:** Loss landscapes in neural networks, non‑convex optimisation, economics.

---

### 03 – Directional Derivatives
**Topic:** Rate of change in arbitrary directions, gradient interpretation.  
**Visualisation:** Paraboloid `z = x² + y²` with point, gradient, and directional derivative vectors.  
**Key Techniques:** Gradient computation, directional derivative formula, vector visualisation.  
**Applications:** Gradient descent, sensitivity analysis, feature importance.

---

### 04 – Gradient Descent
**Topic:** Optimisation algorithm visualisation.  
**Visualisation:** Convex loss surface `z = x² + y²` with descent path, start/end markers.  
**Key Techniques:** Gradient descent algorithm, learning rate, convergence analysis.  
**Applications:** Machine learning training, parameter estimation, numerical optimisation.

---

### 05 – Lagrange Multipliers
**Topic:** Constrained optimisation with equality constraints.  
**Visualisation:** Objective `f(x,y) = xy` with constraint `x + y = 10`; optimal point highlighted.  
**Key Techniques:** Lagrange multiplier method, gradient alignment.  
**Applications:** Resource allocation, economics, regularisation (Lasso/Ridge), physics.

---

### 06 – Taylor Series in 2D
**Topic:** Local approximation using Taylor polynomials.  
**Visualisation:** Gaussian `f(x,y) = e^-(x²+y²)` with first‑ and second‑order approximations.  
**Key Techniques:** Gradient, Hessian, Taylor expansion, error analysis.  
**Applications:** Newton's method, numerical analysis, sensitivity analysis.

---

### 07 – Vector Calculus (Grad, Div, Curl, Integrals)
**Topic:** Comprehensive vector calculus overview.  
**Visualisation:** 
- Gradient as 3D cones on a scalar field.
- Divergence as heatmap over a 2D vector field.
- Curl as heatmap showing rotation.
- Line integral (work) with force field and path.
- Double integral over a disk with surface visualisation.
- Triple integral over a sphere with cross‑section visualisation.  
**Key Techniques:** Numerical integration (`quad`, `dblquad`, `tplquad`), vector field visualisation.  
**Applications:** Fluid dynamics, electromagnetism, physics, advanced ML (neural ODEs).

---

## Setup and Dependencies

### Environment

All notebooks are designed to run in a Conda environment named `ai_env` with **Python 3.10**. 