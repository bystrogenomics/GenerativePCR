# Generative Principal Component Regression via Variational Inference

[![Status](https://img.shields.io/badge/status-complete-success.svg)](#)

This repository contains the finalized and complete official code implementation and reproducibility materials for the paper:

**Generative Principal Component Regression via Variational Inference**

Published in **IEEE Transactions on Signal Processing**.

- **IEEE Xplore:** [ieeexplore.ieee.org/abstract/document/11478741](https://ieeexplore.ieee.org/abstract/document/11478741)
- **DOI:** [10.1109/TSP.2026.3682931](https://doi.org/10.1109/TSP.2026.3682931)
- **arXiv preprint:** [arxiv.org/abs/2409.02327](https://arxiv.org/abs/2409.02327)

---

## Repository Status

This repository is **complete and fully implemented**. All models, algorithms, covariance estimation methods, and corresponding unit tests from the paper have been successfully integrated and verified. No further active development or major feature additions are planned, though maintenance updates may be applied if issues arise.

---

## About the Paper

This paper introduces **generative principal component regression (gPCR)**, a variational-inference method for learning latent variable models that retain information relevant to a supervised outcome.

### Motivation
Standard generative latent variable models, such as probabilistic principal component analysis (PPCA), capture the dominant sources of variation in the data but can easily fail to represent lower-variance directions that are critical for prediction or downstream manipulation. This can make the learned latent space poorly aligned with the scientific or experimental target of interest.

### Solution
gPCR addresses this by using an objective inspired by supervised variational autoencoders (VAEs), specialized to linear generative models. The method produces latent representations that are both generative and predictive. In the paper, the approach is evaluated in simulations and on neuroscience datasets related to stress and social behavior. gPCR improves predictive performance and target selection relative to standard principal component regression (PCR) and supervised VAE baselines.

---

## Requirements

This project requires **Python >= 3.10** and the following dependencies:

*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `scipy`
*   `torch`
*   `tqdm`
*   `matplotlib`
*   `pytest`

For development, additional packages are needed:
*   `ruff >= 0.6.9`
*   `pytest >= 8.3`
*   `pytest-cov >= 5.0`
*   `black == 24.2.0`

---

## Installation

You can install the package in editable mode directly from the repository root:

```bash
# Standard installation
pip install .

# Development installation (includes ruff, black, and testing tools)
pip install -e ".[dev]"
```

---

## Repository Structure

The project layout is structured as follows:

```
├── gpcr/                              # Main package source directory
│   ├── gpcr.py                        # GPCRvi: Supervised PPCA via stochastic variational inference
│   ├── generative_fa.py               # PPCA and FactorAnalysis models fit via SGD in PyTorch
│   ├── ppca_augmented.py              # Supervised, adversarial, and randomized augmented PPCA models
│   ├── covariance_cov_shrinkage.py    # Covariance shrinkage estimators (Ledoit-Wolf)
│   ├── _base.py                       # Abstract base classes and shared factor model methods
│   ├── _base_covariance.py            # Likelihood, entropy, and mutual information computations
│   ├── _covariance_np.py              # Empirical and shrinkage covariance estimators
│   ├── _misc_np.py                    # Helper utilities (missingness classification, math helpers)
│   └── _sherman_woodbury_pt.py        # PyTorch Sherman-Woodbury likelihood utility
│
├── tests/                             # Unit tests
│   └── test__misc_np.py               # Tests for utility functions
│
├── pyproject.toml                     # Package metadata, dependencies, and tools configuration
├── LICENSE                            # License details (MIT)
└── README.md                          # Repository documentation (this file)
```

### Key Components

*   **[gpcr.py](file:///home/austin/Forks/GenerativePCR/gpcr/gpcr.py):** Implements the main class `GPCRvi`, which extends `PPCA`. It uses variational inference to supervise a subset of the latent variables to be predictive of an auxiliary outcome (supporting both classification and regression tasks). It features staged unfreezing of the predictive head, warmup iterations, and proximal/L2-anchoring trust region penalties to stabilize training.
*   **[ppca_augmented.py](file:///home/austin/Forks/GenerativePCR/gpcr/ppca_augmented.py):** Implements standard closed-form augmented PCA/PPCA models (`PPCAadversarial`, `PPCAsupervised`, `PPCASupAdversarial`) and their randomized counterparts (`PPCAadversarialRandomized`, `PPCAsupervisedRandomized`) for high-dimensional scaling.
*   **[generative_fa.py](file:///home/austin/Forks/GenerativePCR/gpcr/generative_fa.py):** PyTorch SGD implementations of standard Probabilistic PCA and Factor Analysis.
*   **[_base.py](file:///home/austin/Forks/GenerativePCR/gpcr/_base.py):** Implements the parent `BaseGaussianFactorModel` class providing core methods like `transform`, `precision`, `entropy`, and scoring/conditional log-likelihood estimation under arbitrary missingness patterns.

---

## Example Usage

Here is a quick example showing how to fit `GPCRvi` to supervised data:

```python
import numpy as np
from gpcr.gpcr import GPCRvi

# 1. Generate synthetic data
rng = np.random.default_rng(42)
n_samples = 200
n_features = 20
X = rng.normal(size=(n_samples, n_features))

# Generate a binary target supervised by a subset of features
true_latent = X[:, 0] + X[:, 1]
probs = 1 / (1 + np.exp(-true_latent))
y = rng.binomial(1, probs)

# Center the input features
X_centered = X - np.mean(X, axis=0)

# 2. Instantiate and fit GPCRvi
model = GPCRvi(
    n_components=3,       # Total latent dimensions
    n_supervised=1,       # First dimension is supervised to predict target
    mu=5.0,               # Strength of supervision loss
    gamma=10.0            # Factor orthogonalization penalty
)

model.fit(X_centered, y, task="classification", progress_bar=True)

# 3. Inspect parameters & transform data
print("Factor Loadings (W) shape:", model.W_.shape)
print("Noise Variance (sigma^2):", model.sigma2_)
print("Predictive Weights (D):", model.d_weights_)

# Project covariates to latent space
latent_representations = model.transform(X_centered)
print("Latent Space representation shape:", latent_representations.shape)
```

---

## Running Tests

To verify the installation and run unit tests, execute:

```bash
PYTHONPATH=. pytest
```

---

## Citation

Please cite the IEEE paper if you use this repository or build on this work:

```bibtex
@article{talbot2026gpcr,
  title   = {Generative Principal Component Regression via Variational Inference},
  author  = {Talbot, Austin and Keller, Corey J. and Carlson, David E. and Kotlar, Alex V.},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2026},
  doi     = {10.1109/TSP.2026.3682931},
  url     = {https://ieeexplore.ieee.org/abstract/document/11478741}
}
```
