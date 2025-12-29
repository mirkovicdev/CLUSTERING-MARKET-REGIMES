"""
Synthetic data generators for testing regime clustering algorithms.

Implements:
1. Geometric Brownian Motion (gBm) with regime switching
2. Merton Jump Diffusion with regime switching

Based on: "Clustering Market Regimes Using the Wasserstein Distance"
by B. Horvath, Z. Issa, and A. Muguruza (2021)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings


@dataclass
class RegimeSwitchingParams:
    """Parameters for regime-switching synthetic data generation."""
    # Number of timesteps per year (252 trading days * 7 hours)
    timesteps_per_year: int = 252 * 7

    # Total years of data
    n_years: int = 20

    # Number of regime changes
    n_regime_changes: int = 10

    # Length of each regime change (in timesteps)
    regime_length: int = 252 * 7 // 2  # Half year

    # Random seed
    random_state: Optional[int] = None


@dataclass
class GBMParams:
    """Parameters for Geometric Brownian Motion."""
    mu: float  # Drift
    sigma: float  # Volatility


@dataclass
class MertonParams:
    """Parameters for Merton Jump Diffusion."""
    mu: float  # Drift
    sigma: float  # Diffusion volatility
    lambda_: float  # Jump intensity
    gamma: float  # Mean of log-jump size
    delta: float  # Std of log-jump size


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Geometric Brownian Motion.

    dS_t = μ S_t dt + σ S_t dW_t

    Log-return distribution:
    ln(S_t) ~ Normal((μ - σ²/2)t, σ²t)

    Args:
        S0: Initial price
        mu: Drift parameter
        sigma: Volatility parameter
        T: Total time
        n_steps: Number of time steps
        random_state: Random seed

    Returns:
        Tuple of (prices, times)
    """
    rng = np.random.default_rng(random_state)
    dt = T / n_steps

    # Generate Brownian increments
    dW = rng.normal(0, np.sqrt(dt), n_steps)

    # Compute log-returns using exact solution
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * dW

    # Cumulative sum to get log-prices
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns)

    prices = np.exp(log_prices)
    times = np.linspace(0, T, n_steps + 1)

    return prices, times


def simulate_merton_jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    lambda_: float,
    gamma: float,
    delta: float,
    T: float,
    n_steps: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Merton Jump Diffusion.

    dS_t = μ S_t dt + σ S_t dW_t + S_{t-} dJ_t

    where J_t = Σ_{j=1}^{N_t} V_j - 1
    with N_t ~ Poisson(λt) and ln(1 + V_j) ~ Normal(γ, δ²)

    From Equation (39-40):
    E[R_t^M] = (μ - σ²/2) + λγ dt
    Var(R_t^M) = (σ² + λ(δ² + γ²)) dt

    Args:
        S0: Initial price
        mu: Drift parameter
        sigma: Diffusion volatility
        lambda_: Jump intensity (Poisson rate)
        gamma: Mean of log-jump size
        delta: Std of log-jump size
        T: Total time
        n_steps: Number of time steps
        random_state: Random seed

    Returns:
        Tuple of (prices, times)
    """
    rng = np.random.default_rng(random_state)
    dt = T / n_steps

    # Initialize arrays
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)

    for t in range(n_steps):
        # Brownian part
        dW = rng.normal(0, np.sqrt(dt))

        # Jump part
        n_jumps = rng.poisson(lambda_ * dt)
        if n_jumps > 0:
            # Log-jump sizes: ln(1 + V_j) ~ Normal(gamma, delta²)
            log_jump_sizes = rng.normal(gamma, delta, n_jumps)
            total_log_jump = np.sum(log_jump_sizes)
        else:
            total_log_jump = 0

        # Update log-price
        # For jump-diffusion: d(ln S) = (μ - σ²/2) dt + σ dW + jump
        log_prices[t + 1] = log_prices[t] + (mu - 0.5 * sigma ** 2) * dt + sigma * dW + total_log_jump

    prices = np.exp(log_prices)
    times = np.linspace(0, T, n_steps + 1)

    return prices, times


def generate_regime_switching_gbm(
    params: RegimeSwitchingParams,
    theta_bull: GBMParams,
    theta_bear: GBMParams,
    S0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Generate regime-switching Geometric Brownian Motion.

    Args:
        params: Regime switching parameters
        theta_bull: GBM parameters for bull regime (normal)
        theta_bear: GBM parameters for bear regime (crisis)
        S0: Initial price

    Returns:
        Tuple of (prices, times, regime_labels, regime_intervals)
        - regime_labels: 0 for bull, 1 for bear
        - regime_intervals: List of (start, end) indices for bear regimes
    """
    rng = np.random.default_rng(params.random_state)

    n_steps = params.timesteps_per_year * params.n_years
    dt = params.n_years / n_steps

    # Generate regime change start points
    # Ensure they don't overlap
    available_starts = list(range(0, n_steps - params.regime_length - params.regime_length, params.regime_length * 2))

    if len(available_starts) < params.n_regime_changes:
        warnings.warn(f"Cannot fit {params.n_regime_changes} regime changes. Using {len(available_starts)}.")
        regime_starts = available_starts
    else:
        regime_starts = sorted(rng.choice(available_starts, size=params.n_regime_changes, replace=False))

    # Create regime label array
    regime_labels = np.zeros(n_steps, dtype=int)
    regime_intervals = []

    for start in regime_starts:
        end = min(start + params.regime_length, n_steps)
        regime_labels[start:end] = 1
        regime_intervals.append((start, end))

    # Generate price path
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)

    for t in range(n_steps):
        if regime_labels[t] == 0:
            # Bull regime
            mu, sigma = theta_bull.mu, theta_bull.sigma
        else:
            # Bear regime
            mu, sigma = theta_bear.mu, theta_bear.sigma

        dW = rng.normal(0, np.sqrt(dt))
        log_prices[t + 1] = log_prices[t] + (mu - 0.5 * sigma ** 2) * dt + sigma * dW

    prices = np.exp(log_prices)
    times = np.linspace(0, params.n_years, n_steps + 1)

    return prices, times, regime_labels, regime_intervals


def generate_regime_switching_merton(
    params: RegimeSwitchingParams,
    theta_bull: MertonParams,
    theta_bear: MertonParams,
    S0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Generate regime-switching Merton Jump Diffusion.

    Args:
        params: Regime switching parameters
        theta_bull: Merton parameters for bull regime
        theta_bear: Merton parameters for bear regime
        S0: Initial price

    Returns:
        Tuple of (prices, times, regime_labels, regime_intervals)
    """
    rng = np.random.default_rng(params.random_state)

    n_steps = params.timesteps_per_year * params.n_years
    dt = params.n_years / n_steps

    # Generate regime change start points
    available_starts = list(range(0, n_steps - params.regime_length - params.regime_length, params.regime_length * 2))

    if len(available_starts) < params.n_regime_changes:
        warnings.warn(f"Cannot fit {params.n_regime_changes} regime changes. Using {len(available_starts)}.")
        regime_starts = available_starts
    else:
        regime_starts = sorted(rng.choice(available_starts, size=params.n_regime_changes, replace=False))

    # Create regime label array
    regime_labels = np.zeros(n_steps, dtype=int)
    regime_intervals = []

    for start in regime_starts:
        end = min(start + params.regime_length, n_steps)
        regime_labels[start:end] = 1
        regime_intervals.append((start, end))

    # Generate price path
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(S0)

    for t in range(n_steps):
        if regime_labels[t] == 0:
            theta = theta_bull
        else:
            theta = theta_bear

        # Brownian part
        dW = rng.normal(0, np.sqrt(dt))

        # Jump part
        n_jumps = rng.poisson(theta.lambda_ * dt)
        if n_jumps > 0:
            log_jump_sizes = rng.normal(theta.gamma, theta.delta, n_jumps)
            total_log_jump = np.sum(log_jump_sizes)
        else:
            total_log_jump = 0

        # Update
        log_prices[t + 1] = log_prices[t] + (theta.mu - 0.5 * theta.sigma ** 2) * dt + theta.sigma * dW + total_log_jump

    prices = np.exp(log_prices)
    times = np.linspace(0, params.n_years, n_steps + 1)

    return prices, times, regime_labels, regime_intervals


def compute_accuracy_scores(
    predicted_labels: np.ndarray,
    true_regime_labels: np.ndarray,
    h1: int,
    h2: int
) -> Tuple[float, float, float]:
    """
    Compute accuracy scores for regime clustering.

    Definition 3.7 from paper:
    - ROFS (Regime-Off Accuracy Score): Accuracy during normal regime
    - RONS (Regime-On Accuracy Score): Accuracy during regime change
    - TA (Total Accuracy)

    Args:
        predicted_labels: Cluster labels from clustering algorithm
        true_regime_labels: True regime labels (0=normal, 1=crisis)
        h1: Window length
        h2: Sliding offset

    Returns:
        Tuple of (total_accuracy, regime_on_accuracy, regime_off_accuracy)
    """
    n_returns = len(true_regime_labels)
    n_windows = len(predicted_labels)

    # Map each return to its window memberships
    return_predictions = []

    for i in range(n_returns):
        # Find which windows contain this return
        window_labels = []
        for w in range(n_windows):
            start = w * h2
            end = start + h1
            if start <= i < end:
                window_labels.append(predicted_labels[w])

        if window_labels:
            # Majority vote for this return
            return_predictions.append(int(np.mean(window_labels) > 0.5))
        else:
            return_predictions.append(-1)  # Not covered

    return_predictions = np.array(return_predictions)

    # Compute accuracies
    # Normal regime: true label 0, prediction should be 0
    normal_mask = (true_regime_labels == 0) & (return_predictions >= 0)
    if np.sum(normal_mask) > 0:
        regime_off_acc = np.mean(return_predictions[normal_mask] == 0)
    else:
        regime_off_acc = 0

    # Crisis regime: true label 1, prediction should be 1
    crisis_mask = (true_regime_labels == 1) & (return_predictions >= 0)
    if np.sum(crisis_mask) > 0:
        regime_on_acc = np.mean(return_predictions[crisis_mask] == 1)
    else:
        regime_on_acc = 0

    # Total accuracy
    valid_mask = return_predictions >= 0
    if np.sum(valid_mask) > 0:
        correct_normal = np.sum((true_regime_labels[valid_mask] == 0) & (return_predictions[valid_mask] == 0))
        correct_crisis = np.sum((true_regime_labels[valid_mask] == 1) & (return_predictions[valid_mask] == 1))
        total_acc = (correct_normal + correct_crisis) / np.sum(valid_mask)
    else:
        total_acc = 0

    return total_acc, regime_on_acc, regime_off_acc


def get_theoretical_moments_gbm(
    theta: GBMParams,
    dt: float
) -> Tuple[float, float]:
    """
    Get theoretical mean and variance of gBm log-returns.

    From Equation (34-35):
    ln(S_t) ~ Normal((μ - σ²/2)dt, σ²dt)

    Args:
        theta: GBM parameters
        dt: Time step

    Returns:
        Tuple of (mean, variance)
    """
    mean = (theta.mu - theta.sigma ** 2 / 2) * dt
    var = theta.sigma ** 2 * dt
    return mean, var


def get_theoretical_moments_merton(
    theta: MertonParams,
    dt: float
) -> Tuple[float, float]:
    """
    Get theoretical mean and variance of Merton log-returns.

    From Equations (39-40):
    E[R_t^M] = (μ - σ²/2) + λγ dt
    Var(R_t^M) = (σ² + λ(δ² + γ²)) dt

    Args:
        theta: Merton parameters
        dt: Time step

    Returns:
        Tuple of (mean, variance)
    """
    mean = (theta.mu - theta.sigma ** 2 / 2) * dt + theta.lambda_ * theta.gamma * dt
    var = (theta.sigma ** 2 + theta.lambda_ * (theta.delta ** 2 + theta.gamma ** 2)) * dt
    return mean, var


# Default parameters from paper
DEFAULT_GBM_BULL = GBMParams(mu=0.02, sigma=0.2)
DEFAULT_GBM_BEAR = GBMParams(mu=-0.02, sigma=0.3)

DEFAULT_MERTON_BULL = MertonParams(mu=0.05, sigma=0.2, lambda_=5, gamma=0.02, delta=0.0125)
DEFAULT_MERTON_BEAR = MertonParams(mu=-0.05, sigma=0.4, lambda_=10, gamma=-0.04, delta=0.1)
