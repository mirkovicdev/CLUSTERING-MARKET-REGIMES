"""
Wasserstein k-means Market Regime Clustering

A Python implementation of the algorithm from:
"Clustering Market Regimes Using the Wasserstein Distance"
by B. Horvath, Z. Issa, and A. Muguruza (2021)

Main Components:
- wasserstein_kmeans: Core clustering algorithms
- synthetic_data: Synthetic data generators
- visualization: Plotting functions
"""

from .wasserstein_kmeans import (
    WassersteinKMeans,
    MomentKMeans,
    compute_log_returns,
    create_sliding_windows,
    wasserstein_distance_1d,
    wasserstein_barycenter_1d,
    compute_mmd_fast,
    compute_self_similarity
)

from .synthetic_data import (
    GBMParams,
    MertonParams,
    RegimeSwitchingParams,
    generate_regime_switching_gbm,
    generate_regime_switching_merton
)

__version__ = "1.0.0"
__author__ = "Based on Horvath, Issa, Muguruza (2021)"
