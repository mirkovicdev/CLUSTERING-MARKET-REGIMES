"""
Visualization functions for market regime clustering.

Reproduces plots from: "Clustering Market Regimes Using the Wasserstein Distance"
by B. Horvath, Z. Issa, and A. Muguruza (2021)

Plots include:
1. Mean-Variance scatter plots (Figure 1)
2. Historical cluster coloring (Figure 2)
3. MMD histograms (Figures 3, 4)
4. Centroid approximation plots (Figures 8, 13)
5. Synthetic path examples (Figures 5, 7, 9, 12)
6. Hyperparameter sensitivity (Figures 14, 15)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Union
from datetime import datetime
import pandas as pd


# ============================================================================
# DARK MODE THEME SETUP
# ============================================================================

# Dark mode color palette
DARK_BG = '#0d1117'
DARK_BG_SECONDARY = '#161b22'
DARK_GRID = '#30363d'
DARK_TEXT = '#e6edf3'
DARK_TEXT_SECONDARY = '#8b949e'

# Vibrant colors for dark mode
CLUSTER_COLORS = ['#00d9ff', '#ff6b6b']  # Cyan for low-vol, Coral-red for high-vol
REGIME_COLORS = ['#58a6ff', '#ff6b6b', '#3fb950', '#a371f7', '#f0883e', '#39d353']

# Set dark mode style globally
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': DARK_BG_SECONDARY,
    'axes.edgecolor': DARK_GRID,
    'axes.labelcolor': DARK_TEXT,
    'axes.titlecolor': DARK_TEXT,
    'text.color': DARK_TEXT,
    'xtick.color': DARK_TEXT_SECONDARY,
    'ytick.color': DARK_TEXT_SECONDARY,
    'grid.color': DARK_GRID,
    'grid.alpha': 0.3,
    'legend.facecolor': DARK_BG_SECONDARY,
    'legend.edgecolor': DARK_GRID,
    'legend.labelcolor': DARK_TEXT,
    'savefig.facecolor': DARK_BG,
    'savefig.edgecolor': DARK_BG,
    'font.family': 'sans-serif',
    'font.size': 11,
})


def compute_distribution_stats(distributions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation for each distribution.

    Projection map: mu -> (sqrt(Var(mu)), E[mu])

    Args:
        distributions: List of empirical distributions

    Returns:
        Tuple of (std_devs, means)
    """
    means = np.array([np.mean(d) for d in distributions])
    stds = np.array([np.std(d) for d in distributions])
    return stds, means


def plot_mean_variance_scatter(
    distributions: List[np.ndarray],
    labels: np.ndarray,
    centroids: List[np.ndarray],
    title: str = "Mean-Variance Scatter Plot",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create mean-variance scatter plot (Figure 1 in paper).

    Args:
        distributions: List of empirical distributions
        labels: Cluster labels
        centroids: Centroid distributions
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute stats for all distributions
    stds, means = compute_distribution_stats(distributions)

    # Plot points by cluster with glow effect
    n_clusters = len(centroids)
    for k in range(n_clusters):
        mask = labels == k
        # Glow effect (larger, more transparent points behind)
        ax.scatter(
            stds[mask], means[mask],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            alpha=0.15,
            s=80,
        )
        # Main points
        ax.scatter(
            stds[mask], means[mask],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            alpha=0.7,
            s=25,
            label=f'Cluster {k+1}',
            edgecolors='none'
        )

    # Plot centroids with glow
    centroid_stds, centroid_means = compute_distribution_stats(centroids)
    for k in range(n_clusters):
        # Glow
        ax.scatter(
            centroid_stds[k], centroid_means[k],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            s=400,
            alpha=0.3,
        )
        # Main centroid marker
        ax.scatter(
            centroid_stds[k], centroid_means[k],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            marker='X',
            s=200,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )

    ax.set_xlabel('Standard Deviation (sigma)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean (mu)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_historical_coloring(
    prices: np.ndarray,
    labels: np.ndarray,
    h1: int,
    h2: int,
    dates: Optional[np.ndarray] = None,
    regime_intervals: Optional[List[Tuple[int, int]]] = None,
    crisis_labels: Optional[Dict[str, Tuple[datetime, datetime]]] = None,
    title: str = "Market Regime Detection",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create historical cluster coloring plot (Figure 2 in paper).

    Colors the price path according to cluster membership of overlapping windows.

    Args:
        prices: Price array
        labels: Cluster labels for each window
        h1: Window length
        h2: Sliding offset
        dates: Optional datetime array
        regime_intervals: List of (start, end) tuples for known regime changes
        crisis_labels: Dict of crisis names to (start_date, end_date)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_prices = len(prices)
    n_windows = len(labels)

    # Create color intensity array based on average cluster membership
    color_values = np.zeros(n_prices - 1)
    counts = np.zeros(n_prices - 1)

    for w in range(n_windows):
        start = w * h2
        end = min(start + h1, n_prices - 1)
        for i in range(start, end):
            if i < n_prices - 1:
                color_values[i] += labels[w]
                counts[i] += 1

    # Normalize
    counts[counts == 0] = 1
    color_values = color_values / counts

    # Create custom colormap for dark mode
    cmap = LinearSegmentedColormap.from_list(
        'regime',
        [CLUSTER_COLORS[0], CLUSTER_COLORS[1]],
        N=256
    )

    # Plot price path with color gradient
    if dates is not None:
        x = dates[:-1]
    else:
        x = np.arange(n_prices - 1)

    # Plot colored segments with glow effect
    for i in range(len(x) - 1):
        color = cmap(color_values[i])
        # Glow
        ax.plot(
            [x[i], x[i+1]],
            [prices[i], prices[i+1]],
            color=color,
            linewidth=4,
            alpha=0.3
        )
        # Main line
        ax.plot(
            [x[i], x[i+1]],
            [prices[i], prices[i+1]],
            color=color,
            linewidth=1.5
        )

    # Add shaded regions for known crisis periods
    if regime_intervals is not None:
        for start, end in regime_intervals:
            if dates is not None:
                ax.axvspan(dates[start], dates[min(end, len(dates)-1)],
                          alpha=0.15, color=CLUSTER_COLORS[1], label='_nolegend_')
            else:
                ax.axvspan(start, end, alpha=0.15, color=CLUSTER_COLORS[1], label='_nolegend_')

    # Add crisis labels if provided
    if crisis_labels is not None:
        for i, (name, (start_date, end_date)) in enumerate(crisis_labels.items()):
            ax.axvspan(start_date, end_date, alpha=0.1, color=REGIME_COLORS[i % len(REGIME_COLORS)])

    # Format x-axis for dates
    if dates is not None and isinstance(dates[0], (datetime, pd.Timestamp, np.datetime64)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    # Add legend
    legend_elements = [
        Patch(facecolor=CLUSTER_COLORS[0], label='Low Volatility (Bull)'),
        Patch(facecolor=CLUSTER_COLORS[1], label='High Volatility (Bear)'),
    ]
    if regime_intervals is not None:
        legend_elements.append(Patch(facecolor=CLUSTER_COLORS[1], alpha=0.3, label='Known Regime Switch'))

    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_mmd_histogram_between_clusters(
    mmd_wk: np.ndarray,
    mmd_mk: np.ndarray,
    title: str = "Between-Cluster MMD Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot between-cluster MMD histogram comparing WK-means and MK-means (Figure 3).

    Args:
        mmd_wk: MMD scores from WK-means
        mmd_mk: MMD scores from MK-means
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    bins = np.linspace(0, max(np.max(mmd_wk), np.max(mmd_mk)) * 1.1, 50)

    ax.hist(mmd_wk, bins=bins, alpha=0.7, label='Wasserstein', color=CLUSTER_COLORS[0], density=True, edgecolor='white', linewidth=0.5)
    ax.hist(mmd_mk, bins=bins, alpha=0.7, label='Moments', color=CLUSTER_COLORS[1], density=True, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('MMD squared', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_mmd_histogram_within_clusters(
    mmd_wk_c1: np.ndarray,
    mmd_wk_c2: np.ndarray,
    mmd_mk_c1: np.ndarray,
    mmd_mk_c2: np.ndarray,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot within-cluster MMD histograms (Figure 4).

    Args:
        mmd_wk_c1: MMD scores for WK-means cluster 1
        mmd_wk_c2: MMD scores for WK-means cluster 2
        mmd_mk_c1: MMD scores for MK-means cluster 1
        mmd_mk_c2: MMD scores for MK-means cluster 2
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Cluster 1
    ax1 = axes[0]
    max_val = max(np.max(mmd_wk_c1), np.max(mmd_mk_c1)) * 1.1
    bins = np.linspace(0, max_val, 50)

    ax1.hist(mmd_wk_c1, bins=bins, alpha=0.7, label='Wasserstein', color=CLUSTER_COLORS[0], density=True, edgecolor='white', linewidth=0.5)
    ax1.hist(mmd_mk_c1, bins=bins, alpha=0.7, label='Moments', color=CLUSTER_COLORS[1], density=True, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('MMD squared (C1)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Within-Cluster MMD (Cluster 1)', fontsize=12, fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # Cluster 2
    ax2 = axes[1]
    max_val = max(np.max(mmd_wk_c2), np.max(mmd_mk_c2)) * 1.1
    bins = np.linspace(0, max_val, 50)

    ax2.hist(mmd_wk_c2, bins=bins, alpha=0.7, label='Wasserstein', color=CLUSTER_COLORS[0], density=True, edgecolor='white', linewidth=0.5)
    ax2.hist(mmd_mk_c2, bins=bins, alpha=0.7, label='Moments', color=CLUSTER_COLORS[1], density=True, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('MMD squared (C2)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Within-Cluster MMD (Cluster 2)', fontsize=12, fontweight='bold')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_synthetic_path_with_regimes(
    prices: np.ndarray,
    times: np.ndarray,
    regime_labels: np.ndarray,
    title: str = "Synthetic Price Path with Regime Changes",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot synthetic price path with regime changes highlighted (Figure 5, 9).

    Args:
        prices: Price array
        times: Time array
        regime_labels: True regime labels (0=normal, 1=crisis)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Highlight regime change periods first (so they're behind)
    in_regime = False
    start_idx = 0

    for i in range(len(regime_labels)):
        if regime_labels[i] == 1 and not in_regime:
            start_idx = i
            in_regime = True
        elif regime_labels[i] == 0 and in_regime:
            ax.axvspan(times[start_idx], times[i], alpha=0.2, color=CLUSTER_COLORS[1], label='_nolegend_')
            in_regime = False

    if in_regime:
        ax.axvspan(times[start_idx], times[-1], alpha=0.2, color=CLUSTER_COLORS[1], label='_nolegend_')

    # Plot main price path with glow
    ax.plot(times[:-1], prices[:-1], color=CLUSTER_COLORS[0], linewidth=3, alpha=0.3)
    ax.plot(times[:-1], prices[:-1], color=CLUSTER_COLORS[0], linewidth=1.2, label='Price Path')

    ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=CLUSTER_COLORS[0], linewidth=2, label='Price Path'),
        Patch(facecolor=CLUSTER_COLORS[1], alpha=0.3, label='High-Vol Regime')
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_log_returns_with_regimes(
    log_returns: np.ndarray,
    times: np.ndarray,
    regime_labels: np.ndarray,
    title: str = "Log Returns with Regime Changes",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot log returns with regime changes highlighted.

    Args:
        log_returns: Log return array
        times: Time array
        regime_labels: True regime labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Highlight regime change periods first
    in_regime = False
    start_idx = 0

    for i in range(len(regime_labels)):
        if regime_labels[i] == 1 and not in_regime:
            start_idx = i
            in_regime = True
        elif regime_labels[i] == 0 and in_regime:
            ax.axvspan(times[start_idx], times[i], alpha=0.2, color=CLUSTER_COLORS[1], label='_nolegend_')
            in_regime = False

    if in_regime:
        ax.axvspan(times[start_idx], times[-1], alpha=0.2, color=CLUSTER_COLORS[1], label='_nolegend_')

    # Plot returns with glow
    ax.plot(times[:-1], log_returns, color=CLUSTER_COLORS[0], linewidth=2, alpha=0.3)
    ax.plot(times[:-1], log_returns, color=CLUSTER_COLORS[0], linewidth=0.6, label='Log Returns')

    ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Log Return', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    legend_elements = [
        plt.Line2D([0], [0], color=CLUSTER_COLORS[0], linewidth=1, label='Log Returns'),
        Patch(facecolor=CLUSTER_COLORS[1], alpha=0.3, label='High-Vol Regime')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_centroid_approximation(
    distributions: List[np.ndarray],
    labels: np.ndarray,
    centroids: List[np.ndarray],
    theoretical_means: Tuple[float, float],
    theoretical_vars: Tuple[float, float],
    algorithm_name: str = "WK-means",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of segment means/variances with theoretical values (Figure 8, 13).

    Args:
        distributions: List of empirical distributions
        labels: Cluster labels
        centroids: Centroids
        theoretical_means: (mean_bull, mean_bear)
        theoretical_vars: (var_bull, var_bear)
        algorithm_name: Name of algorithm
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Compute stats
    means = np.array([np.mean(d) for d in distributions])
    variances = np.array([np.var(d) for d in distributions])
    centroid_means = [np.mean(c) for c in centroids]
    centroid_vars = [np.var(c) for c in centroids]

    # Plot means
    ax1 = axes[0]
    ax1.hist(means, bins=50, alpha=0.7, color='#8b949e', density=True, edgecolor='white', linewidth=0.5)
    ax1.axvline(theoretical_means[0], color=CLUSTER_COLORS[0], linestyle='-', linewidth=2.5, label='Theoretical Bull')
    ax1.axvline(theoretical_means[1], color=CLUSTER_COLORS[1], linestyle='-', linewidth=2.5, label='Theoretical Bear')
    ax1.axvline(centroid_means[0], color='#3fb950', linestyle='--', linewidth=2, label='Centroid 1')
    ax1.axvline(centroid_means[1], color='#f0883e', linestyle='--', linewidth=2, label='Centroid 2')
    ax1.set_xlabel('Mean (mu)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribution of Segment Means - {algorithm_name}', fontsize=12, fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # Plot variances
    ax2 = axes[1]
    ax2.hist(variances, bins=50, alpha=0.7, color='#8b949e', density=True, edgecolor='white', linewidth=0.5)
    ax2.axvline(theoretical_vars[0], color=CLUSTER_COLORS[0], linestyle='-', linewidth=2.5, label='Theoretical Bull')
    ax2.axvline(theoretical_vars[1], color=CLUSTER_COLORS[1], linestyle='-', linewidth=2.5, label='Theoretical Bear')
    ax2.axvline(centroid_vars[0], color='#3fb950', linestyle='--', linewidth=2, label='Centroid 1')
    ax2.axvline(centroid_vars[1], color='#f0883e', linestyle='--', linewidth=2, label='Centroid 2')
    ax2.set_xlabel('Variance (sigma squared)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribution of Segment Variances - {algorithm_name}', fontsize=12, fontweight='bold')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_skew_kurtosis(
    distributions: List[np.ndarray],
    labels: np.ndarray,
    centroids: List[np.ndarray],
    title: str = "Skewness-Kurtosis Scatter Plot",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distributions in skew-kurtosis space (Figure 11).

    Args:
        distributions: List of empirical distributions
        labels: Cluster labels
        centroids: Centroids
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=figsize)

    # Compute skew and kurtosis
    skews = np.array([sp_stats.skew(d) for d in distributions])
    kurts = np.array([sp_stats.kurtosis(d) for d in distributions])

    # Plot by cluster with glow
    n_clusters = len(centroids)
    for k in range(n_clusters):
        mask = labels == k
        # Glow
        ax.scatter(
            skews[mask], kurts[mask],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            alpha=0.15,
            s=80,
        )
        # Main points
        ax.scatter(
            skews[mask], kurts[mask],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            alpha=0.7,
            s=25,
            label=f'Cluster {k+1}',
            edgecolors='none'
        )

    # Plot centroids
    centroid_skews = [sp_stats.skew(c) for c in centroids]
    centroid_kurts = [sp_stats.kurtosis(c) for c in centroids]

    for k in range(n_clusters):
        # Glow
        ax.scatter(
            centroid_skews[k], centroid_kurts[k],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            s=400,
            alpha=0.3,
        )
        # Main centroid
        ax.scatter(
            centroid_skews[k], centroid_kurts[k],
            c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
            marker='X',
            s=200,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )

    ax.set_xlabel('Skewness', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kurtosis', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_accuracy_vs_window_length(
    window_lengths: np.ndarray,
    total_accuracies: np.ndarray,
    regime_on_accuracies: np.ndarray,
    regime_off_accuracies: np.ndarray,
    total_ci: Optional[np.ndarray] = None,
    regime_on_ci: Optional[np.ndarray] = None,
    regime_off_ci: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot accuracy scores vs window length (Figure 15).

    Args:
        window_lengths: Array of window lengths
        total_accuracies: Total accuracy for each window length
        regime_on_accuracies: Regime-on accuracy for each window length
        regime_off_accuracies: Regime-off accuracy for each window length
        total_ci: 95% CI for total accuracy (optional)
        regime_on_ci: 95% CI for regime-on accuracy (optional)
        regime_off_ci: 95% CI for regime-off accuracy (optional)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = [CLUSTER_COLORS[0], CLUSTER_COLORS[1], '#3fb950']

    # Plot means with glow
    for data, color, label in [
        (total_accuracies, colors[0], 'Total Accuracy'),
        (regime_on_accuracies, colors[1], 'Regime-On Accuracy'),
        (regime_off_accuracies, colors[2], 'Regime-Off Accuracy')
    ]:
        ax.plot(window_lengths, data, color=color, linewidth=4, alpha=0.3)
        ax.plot(window_lengths, data, color=color, linewidth=2, label=label)

    # Plot confidence intervals if provided
    if total_ci is not None:
        ax.fill_between(window_lengths, total_accuracies - total_ci, total_accuracies + total_ci,
                       alpha=0.15, color=colors[0])
    if regime_on_ci is not None:
        ax.fill_between(window_lengths, regime_on_accuracies - regime_on_ci,
                       regime_on_accuracies + regime_on_ci, alpha=0.15, color=colors[1])
    if regime_off_ci is not None:
        ax.fill_between(window_lengths, regime_off_accuracies - regime_off_ci,
                       regime_off_accuracies + regime_off_ci, alpha=0.15, color=colors[2])

    ax.set_xlabel('Window Length (h1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Window Length on Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_comparison_figure(
    wk_fig_func,
    mk_fig_func,
    wk_kwargs: dict,
    mk_kwargs: dict,
    titles: Tuple[str, str] = ("MK-means", "WK-means"),
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create side-by-side comparison figure for two algorithms.

    Args:
        wk_fig_func: Function to create WK-means subplot
        mk_fig_func: Function to create MK-means subplot
        wk_kwargs: Kwargs for WK-means function
        mk_kwargs: Kwargs for MK-means function
        titles: Tuple of subplot titles
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # MK-means
    plt.sca(axes[0])
    mk_fig_func(**mk_kwargs, title=titles[0])
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold')

    # WK-means
    plt.sca(axes[1])
    wk_fig_func(**wk_kwargs, title=titles[1])
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def create_regime_animation(
    prices: np.ndarray,
    labels: np.ndarray,
    h1: int,
    h2: int,
    dates: Optional[np.ndarray] = None,
    title: str = "Market Regime Detection",
    figsize: Tuple[int, int] = (14, 8),
    fps: int = 30,
    duration_seconds: int = 15,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Create an animated visualization of regime detection over time.

    The animation shows the price path being drawn progressively with
    regime coloring revealing as the algorithm processes each window.

    Args:
        prices: Price array
        labels: Cluster labels for each window
        h1: Window length
        h2: Sliding offset
        dates: Optional datetime array
        title: Plot title
        figsize: Figure size
        fps: Frames per second
        duration_seconds: Total animation duration
        save_path: Path to save animation (should be .mp4 or .gif)

    Returns:
        matplotlib animation object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_prices = len(prices)
    n_windows = len(labels)
    total_frames = fps * duration_seconds

    # Compute color values for each price point
    color_values = np.zeros(n_prices - 1)
    counts = np.zeros(n_prices - 1)

    for w in range(n_windows):
        start = w * h2
        end = min(start + h1, n_prices - 1)
        for i in range(start, end):
            if i < n_prices - 1:
                color_values[i] += labels[w]
                counts[i] += 1

    counts[counts == 0] = 1
    color_values = color_values / counts

    # Create colormap
    cmap = LinearSegmentedColormap.from_list(
        'regime',
        [CLUSTER_COLORS[0], CLUSTER_COLORS[1]],
        N=256
    )

    # Prepare x-axis
    if dates is not None:
        x = dates[:-1]
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        x = np.arange(n_prices - 1)

    # Set up the plot
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.min(prices) * 0.95, np.max(prices) * 1.05)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        Patch(facecolor=CLUSTER_COLORS[0], label='Low Volatility (Bull)'),
        Patch(facecolor=CLUSTER_COLORS[1], label='High Volatility (Bear)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    # Line collection for the path
    lines = []
    glows = []

    def init():
        return []

    def animate(frame):
        # Calculate how many points to show
        progress = frame / total_frames
        n_points = int(progress * (len(x) - 1))

        # Clear previous lines
        for line in lines:
            line.remove()
        for glow in glows:
            glow.remove()
        lines.clear()
        glows.clear()

        # Draw the path up to current point
        for i in range(n_points):
            color = cmap(color_values[i])
            # Glow effect
            glow, = ax.plot(
                [x[i], x[i+1]],
                [prices[i], prices[i+1]],
                color=color,
                linewidth=4,
                alpha=0.3
            )
            glows.append(glow)
            # Main line
            line, = ax.plot(
                [x[i], x[i+1]],
                [prices[i], prices[i+1]],
                color=color,
                linewidth=1.5
            )
            lines.append(line)

        # Add a "current position" marker
        if n_points > 0 and n_points < len(prices) - 1:
            marker, = ax.plot(x[n_points], prices[n_points], 'o',
                            color='white', markersize=8, zorder=10)
            lines.append(marker)
            # Pulse effect
            pulse, = ax.plot(x[n_points], prices[n_points], 'o',
                           color='white', markersize=15, alpha=0.3, zorder=9)
            lines.append(pulse)

        return lines + glows

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=True
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps, dpi=100)
        else:
            # For MP4, need ffmpeg
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(save_path, writer=writer, dpi=100)
            except Exception as e:
                print(f"Could not save as MP4 (ffmpeg required). Saving as GIF instead.")
                gif_path = save_path.rsplit('.', 1)[0] + '.gif'
                anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
        print(f"Animation saved!")

    return anim


def create_clustering_animation(
    distributions: List[np.ndarray],
    labels: np.ndarray,
    centroids: List[np.ndarray],
    title: str = "Wasserstein K-Means Clustering",
    figsize: Tuple[int, int] = (10, 8),
    fps: int = 30,
    duration_seconds: int = 10,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Create an animated scatter plot showing points being clustered.

    Points appear one by one and get assigned to clusters with a
    satisfying visual effect.

    Args:
        distributions: List of empirical distributions
        labels: Final cluster labels
        centroids: Final centroids
        title: Plot title
        figsize: Figure size
        fps: Frames per second
        duration_seconds: Total animation duration
        save_path: Path to save animation

    Returns:
        matplotlib animation object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute stats
    stds, means = compute_distribution_stats(distributions)
    centroid_stds, centroid_means = compute_distribution_stats(centroids)

    total_frames = fps * duration_seconds
    n_points = len(distributions)

    # Set up plot
    margin = 0.1
    ax.set_xlim(np.min(stds) - margin * np.ptp(stds), np.max(stds) + margin * np.ptp(stds))
    ax.set_ylim(np.min(means) - margin * np.ptp(means), np.max(means) + margin * np.ptp(means))
    ax.set_xlabel('Standard Deviation (sigma)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean (mu)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    # Pre-plot centroids (faded, will become solid at end)
    centroid_plots = []
    for k in range(len(centroids)):
        # Centroid glow
        glow = ax.scatter(
            centroid_stds[k], centroid_means[k],
            c=CLUSTER_COLORS[k],
            s=400,
            alpha=0.1,
        )
        # Centroid marker
        marker = ax.scatter(
            centroid_stds[k], centroid_means[k],
            c=CLUSTER_COLORS[k],
            marker='X',
            s=200,
            alpha=0.3,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
        centroid_plots.append((glow, marker))

    scatter_artists = []

    def init():
        return []

    def animate(frame):
        progress = frame / total_frames
        n_shown = int(progress * n_points)

        # Remove old scatter plots
        for artist in scatter_artists:
            artist.remove()
        scatter_artists.clear()

        # Draw points up to current
        for i in range(n_shown):
            k = labels[i]
            # Glow
            glow = ax.scatter(
                stds[i], means[i],
                c=CLUSTER_COLORS[k],
                alpha=0.15,
                s=80,
            )
            scatter_artists.append(glow)
            # Main point
            point = ax.scatter(
                stds[i], means[i],
                c=CLUSTER_COLORS[k],
                alpha=0.7,
                s=25,
                edgecolors='none'
            )
            scatter_artists.append(point)

        # Update centroid opacity based on progress
        for k, (glow, marker) in enumerate(centroid_plots):
            glow.set_alpha(0.1 + 0.2 * progress)
            marker.set_alpha(0.3 + 0.7 * progress)

        return scatter_artists

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=True
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps, dpi=100)
        else:
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(save_path, writer=writer, dpi=100)
            except Exception:
                gif_path = save_path.rsplit('.', 1)[0] + '.gif'
                anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
        print(f"Animation saved!")

    return anim
