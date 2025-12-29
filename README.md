# Wasserstein K-Means Market Regime Clustering

A Python implementation of the unsupervised market regime detection algorithm from the paper:

> **"Clustering Market Regimes Using the Wasserstein Distance"**
> B. Horvath, Z. Issa, and A. Muguruza (2021)
> [arXiv:2110.11848](https://arxiv.org/abs/2110.11848)

This repository reproduces the key results from the paper, including real data analysis on SPY and synthetic experiments with Geometric Brownian Motion and Merton Jump Diffusion processes.

## Overview

Traditional clustering methods like k-means operate on finite-dimensional feature vectors. This implementation clusters **probability distributions** directly using the Wasserstein distance, enabling more robust detection of market regimes without relying solely on simple moment statistics.

### Key Features

- **Wasserstein K-Means (WK-means)**: Clusters empirical distributions using the p-Wasserstein distance
- **Moment K-Means (MK-means)**: Benchmark algorithm using statistical moments
- **Synthetic Data Generation**: Regime-switching GBM and Merton jump diffusion
- **Validation Metrics**: Maximum Mean Discrepancy (MMD), self-similarity scores
- **Visualization**: Dark-themed plots and animations for regime detection

## Results

The algorithm successfully identifies known market crisis periods:
- 2008-2009: Global Financial Crisis
- 2010-2011: European Debt Crisis
- 2015-2016: Chinese Stock Market Crash
- 2018: Late-year Bear Market
- 2020: COVID-19 Crash

On synthetic data with known ground truth:
| Data Type | WK-means Accuracy | MK-means Accuracy |
|-----------|-------------------|-------------------|
| GBM | ~82% | ~92% |
| Merton Jump | ~99% | ~75% |

WK-means significantly outperforms on non-Gaussian data (Merton), demonstrating its ability to capture higher-order distributional features beyond mean and variance.

## Installation

```bash
git clone https://github.com/yourusername/regime_clustering.git
cd regime_clustering
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- POT (Python Optimal Transport)
- yfinance (optional, for downloading data)

## Usage

### Run All Experiments

```bash
# Full run (takes ~5-10 minutes)
python run_all.py

# Quick demonstration (fewer iterations)
python run_all.py --quick
```

### Run Individual Components

```bash
# Real data analysis only (SPY)
python main_real_data.py

# Synthetic experiments only
python main_synthetic.py
```

### View Generated Figures

```bash
# Display all figures in a grid
python view_figures.py

# Display in pages of 9
python view_figures.py --pages
```

## Project Structure

```
regime_clustering/
├── wasserstein_kmeans.py   # Core algorithms (WK-means, MK-means, MMD)
├── synthetic_data.py       # GBM and Merton data generators
├── visualization.py        # Plotting functions and animations
├── main_real_data.py       # SPY analysis script
├── main_synthetic.py       # Synthetic experiments
├── run_all.py              # Master script
├── view_figures.py         # Figure viewer utility
├── requirements.txt        # Dependencies
└── figures/                # Generated plots and animations
```

## Algorithm

The Wasserstein K-Means algorithm:

1. **Stream Lift**: Convert price series to overlapping windows of log returns
2. **Empirical Measures**: Treat each window as an empirical probability distribution
3. **Wasserstein Distance**: Measure dissimilarity between distributions using optimal transport
4. **K-Means Clustering**: Iteratively assign distributions to clusters and update centroids (Wasserstein barycenters)

## Key Concepts

### Wasserstein Distance (1D)

For empirical measures with sorted atoms:
```
W_p(μ, ν)^p = (1/N) Σ |α_i - β_i|^p
```

### Wasserstein Barycenter

For p=1: Element-wise median of sorted quantiles
For p=2: Element-wise mean of sorted quantiles

### Maximum Mean Discrepancy (MMD)

Used for cluster validation:
```
MMD²(P, Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{horvath2021clustering,
  title={Clustering Market Regimes Using the Wasserstein Distance},
  author={Horvath, Blanka and Issa, Zacharia and Muguruza, Aitor},
  journal={arXiv preprint arXiv:2110.11848},
  year={2021}
}
```

## License

This implementation is provided for educational and research purposes. The methodology belongs to the original authors.

## Acknowledgments

- Original paper authors: B. Horvath, Z. Issa, and A. Muguruza
- SPY data sourced from Yahoo Finance
