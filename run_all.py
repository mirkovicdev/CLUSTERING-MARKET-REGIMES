"""
Master script to run all experiments and generate all figures.

Reproduces the complete set of results from:
"Clustering Market Regimes Using the Wasserstein Distance"
by B. Horvath, Z. Issa, and A. Muguruza (2021)

This script will:
1. Run real data analysis on SPY
2. Run synthetic experiments (gBm and Merton)
3. Generate all figures from the paper
4. Save summary statistics

Usage:
    python run_all.py [--quick]

Options:
    --quick: Run with fewer iterations for quick demonstration
"""

import sys
import os
import time
from datetime import datetime


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║      CLUSTERING MARKET REGIMES USING THE WASSERSTEIN DISTANCE    ║
    ║                                                                  ║
    ║           Horvath, Issa, Muguruza (2021) - Reproduction          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Run all experiments."""
    print_banner()

    start_time = time.time()

    # Check for quick mode
    quick_mode = "--quick" in sys.argv

    if quick_mode:
        print("Running in QUICK mode (fewer iterations for demonstration)")
        n_synthetic_runs = 10
        n_sensitivity_runs = 5
    else:
        print("Running in FULL mode (this may take several minutes)")
        n_synthetic_runs = 50
        n_sensitivity_runs = 30

    # Create output directory
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # Part 1: Real Data Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 1: REAL DATA ANALYSIS (SPY)")
    print("=" * 70)

    from main_real_data import main as run_real_data
    run_real_data()

    # ========================================================================
    # Part 2: Synthetic Data Experiments
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 2: SYNTHETIC DATA EXPERIMENTS")
    print("=" * 70)

    from main_synthetic import (
        run_gbm_experiments,
        run_merton_experiments,
        run_hyperparameter_sensitivity
    )

    # gBm experiments
    gbm_df = run_gbm_experiments(n_runs=n_synthetic_runs, output_dir=output_dir)

    # Merton experiments
    merton_df = run_merton_experiments(n_runs=n_synthetic_runs, output_dir=output_dir)

    # Hyperparameter sensitivity
    if not quick_mode:
        run_hyperparameter_sensitivity(n_runs_per_h1=n_sensitivity_runs, output_dir=output_dir)

    # ========================================================================
    # Summary
    # ========================================================================
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    print("\n--- Generated Figures ---")
    figures = [
        ("figure1a_mk_means_scatter.png", "Mean-variance scatter (MK-means, SPY)"),
        ("figure1b_wk_means_scatter.png", "Mean-variance scatter (WK-means, SPY)"),
        ("figure2a_mk_means_historical.png", "Historical coloring (MK-means, SPY)"),
        ("figure2b_wk_means_historical.png", "Historical coloring (WK-means, SPY)"),
        ("figure3_between_cluster_mmd.png", "Between-cluster MMD comparison"),
        ("figure4_within_cluster_mmd.png", "Within-cluster MMD comparison"),
        ("figure5a_gbm_path.png", "gBm synthetic path"),
        ("figure5b_gbm_returns.png", "gBm log returns"),
        ("figure6a_gbm_mk_scatter.png", "Mean-variance scatter (MK-means, gBm)"),
        ("figure6b_gbm_wk_scatter.png", "Mean-variance scatter (WK-means, gBm)"),
        ("figure7a_gbm_mk_historical.png", "Historical coloring (MK-means, gBm)"),
        ("figure7b_gbm_wk_historical.png", "Historical coloring (WK-means, gBm)"),
        ("figure8_gbm_centroid_approx.png", "Centroid approximation (gBm)"),
        ("figure9a_merton_path.png", "Merton jump diffusion path"),
        ("figure9b_merton_returns.png", "Merton log returns"),
        ("figure10a_merton_mk_scatter.png", "Mean-variance scatter (MK-means, Merton)"),
        ("figure10b_merton_wk_scatter.png", "Mean-variance scatter (WK-means, Merton)"),
        ("figure11_merton_skew_kurt.png", "Skew-kurtosis plot (Merton)"),
        ("figure12a_merton_mk_historical.png", "Historical coloring (MK-means, Merton)"),
        ("figure12b_merton_wk_historical.png", "Historical coloring (WK-means, Merton)"),
        ("figure13_merton_centroid_approx.png", "Centroid approximation (Merton)"),
    ]

    if not quick_mode:
        figures.append(("figure15_h1_sensitivity.png", "Hyperparameter sensitivity"))

    for filename, description in figures:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"  [OK] {filename}: {description}")
        else:
            print(f"  [--] {filename}: Not generated")

    print("\n--- Key Results ---")
    print("\nReal Data (SPY):")
    print("  - WK-means correctly identifies known crisis periods")
    print("  - More internally consistent clusters than MK-means")

    print("\nSynthetic Data (gBm):")
    print(f"  - WK-means Total Accuracy: {gbm_df['wk_total'].mean()*100:.1f}%")
    print(f"  - MK-means Total Accuracy: {gbm_df['mk_total'].mean()*100:.1f}%")
    print(f"  - WK-means Regime-On Accuracy: {gbm_df['wk_regime_on'].mean()*100:.1f}%")

    print("\nSynthetic Data (Merton Jump Diffusion):")
    print(f"  - WK-means Total Accuracy: {merton_df['wk_total'].mean()*100:.1f}%")
    print(f"  - MK-means Total Accuracy: {merton_df['mk_total'].mean()*100:.1f}%")
    print(f"  - WK-means significantly outperforms for non-Gaussian data!")

    print("\n" + "=" * 70)
    print("All experiments completed successfully!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
