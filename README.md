# Generative Principal Component Regression via Variational Inference

This repository will contain the code and reproducibility materials for the paper:

**Generative Principal Component Regression via Variational Inference**

Published in **IEEE Transactions on Signal Processing**.

- IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11478741
- DOI: https://doi.org/10.1109/TSP.2026.3682931
- arXiv preprint: https://arxiv.org/abs/2409.02327

## Repository status

This repository is currently under construction.

The completed public version of the repository is expected to be available by **May 9, 2026**. Until then, files, APIs, scripts, and directory structure may change without notice.

## About the paper

This paper introduces **generative principal component regression**, abbreviated **gPCR**, a variational-inference method for learning latent variable models that retain information relevant to a supervised outcome.

The motivation is that standard generative latent variable models, such as probabilistic principal component analysis, can capture dominant sources of variation in the data while failing to represent lower-variance directions that are important for prediction or downstream manipulation. This can make the learned latent space poorly aligned with the scientific or experimental target of interest.

gPCR addresses this by using an objective inspired by supervised variational autoencoders, but specialized to linear generative models. The method is designed to produce latent representations that are both generative and predictive. In the paper, the approach is evaluated in simulation and in neuroscience datasets related to stress and social behavior, where it improves predictive performance and target selection relative to standard principal component regression and supervised variational autoencoder baselines.

## Planned repository contents

The final repository will include:

- Source code for fitting gPCR models
- Baseline implementations used in the paper
- Simulation scripts
- Figure-generation scripts
- Installation and environment instructions
- Example commands for running key experiments
- Documentation describing expected inputs and outputs

## Current use

At this stage, this repository should be treated as a placeholder. The code and documentation are not yet finalized, and results should not be assumed to be reproducible until the May 9 release.

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
