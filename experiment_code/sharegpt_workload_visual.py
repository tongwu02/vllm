#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from pathlib import Path

# ================= Configuration =================
INPUT_FILE = Path(__file__).parent / "milestone2_task3_results_complete.json"
OUTPUT_DIR = Path(__file__).parent / "plots_milestone2_final"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
plt.rcParams['font.family'] = 'sans-serif'


# =======================================

def print_result_table(sweep_data):
    """Print formatted experiment results table"""
    print("\n" + "=" * 120)
    print(
        f"{'BS':<5} | {'BN':<8} | {'Policy':<8} | {'Single HR':<10} | {'Multi HR':<10} | {'Improvement':<12} | {'Avg Latency':<12}")
    print("-" * 120)

    # Sort by config for neatness
    sorted_data = sorted(sweep_data, key=lambda x: (x['config']['block_size'], x['config']['block_number']))

    for item in sorted_data:
        c = item['config']
        s_hr = item['single_turn']['hit_rate']
        m_hr = item['multi_turn']['hit_rate']
        imp = item['improvement']
        # Handle cases where avg_latency might be missing in older JSON versions
        lat = item.get('avg_latency', 0) * 1000  # Convert to ms

        print(f"{c['block_size']:<5} | {c['block_number']:<8} | {c['eviction_policy']:<8} | "
              f"{s_hr:<10.2%} | {m_hr:<10.2%} | {imp:<+12.2%} | {lat:<10.2f} ms")

    print("-" * 120 + "\n")


def plot_1_hit_rate_trend(sweep_data):
    """Plot 1: Hit Rate vs Capacity (Block Size Comparison)"""
    rows = []
    for item in sweep_data:
        row = item['config'].copy()
        row['hit_rate'] = item['multi_turn']['hit_rate']
        row['block_size_str'] = f"BS={row['block_size']}"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Aggregate different Policies (since Policy makes no difference in this task)
    df_agg = df.groupby(['block_size_str', 'block_number'])['hit_rate'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df_agg,
        x="block_number",
        y="hit_rate",
        hue="block_size_str",
        style="block_size_str",
        markers=True,
        dashes=False,
        linewidth=3,
        markersize=10,
        palette="viridis"
    )

    ax.set_title("Impact of Cache Capacity & Block Size on Hit Rate", fontweight='bold', pad=15)
    ax.set_xlabel("Cache Capacity (Block Number)")
    ax.set_ylabel("Hit Rate (Multi-Turn)")
    ax.set_xscale('log', base=2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    # Grid and layout
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_hit_rate_capacity.png", dpi=300)
    print(f"✓ Saved Plot 1: Hit Rate Trend")
    plt.close()


def plot_2_shared_fraction_cdf(fractions):
    """Plot 2: Shared Fraction CDF (How many requests benefited from sharing)"""
    fractions = np.sort(fractions)
    yvals = np.arange(len(fractions)) / float(len(fractions) - 1)

    plt.figure(figsize=(8, 6))
    plt.plot(fractions, yvals, color="#2ecc71", linewidth=3)
    plt.fill_between(fractions, yvals, color="#2ecc71", alpha=0.3)

    plt.title("CDF: Fraction of Shared Tokens per Request", fontweight='bold', pad=15)
    plt.xlabel("Fraction Shared (Cached / Total Length)")
    plt.ylabel("Cumulative Probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add text annotation
    median_share = np.median(fractions)
    plt.text(0.05, 0.8, f"Median Share: {median_share:.1%}",
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_shared_fraction_cdf.png", dpi=300)
    print(f"✓ Saved Plot 2: Shared Fraction CDF")
    plt.close()


def plot_3_block_hits_histogram(hits):
    """Plot 3: Block Hits Distribution (Long-tail distribution)"""
    plt.figure(figsize=(8, 6))

    # Use Log-Log Scale to show long-tail effect
    ax = sns.histplot(hits, log_scale=(True, True), color="#3498db", bins=30, kde=False)

    plt.title("Distribution of Block Access Frequency", fontweight='bold', pad=15)
    plt.xlabel("Number of Hits (Log Scale)")
    plt.ylabel("Count of Blocks (Log Scale)")
    plt.grid(True, which="major", linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_block_hits_dist.png", dpi=300)
    print(f"✓ Saved Plot 3: Block Hits Histogram")
    plt.close()


def plot_4_reuse_gap_cdf(gaps):
    """Plot 4: Reuse Gap CDF (Temporal Locality)"""
    if not gaps:
        print("⚠ No reuse gaps found (Cache might be too small or single turn). Skipping Plot 4.")
        return

    gaps = np.sort(gaps)
    yvals = np.arange(len(gaps)) / float(len(gaps) - 1)

    plt.figure(figsize=(8, 6))
    plt.plot(gaps, yvals, color="#e74c3c", linewidth=3)

    plt.title("CDF: Block Reuse Distance (Temporal Locality)", fontweight='bold', pad=15)
    plt.xlabel("Reuse Gap (Number of Intervening Requests)")
    plt.ylabel("Cumulative Probability")
    plt.xscale('log')  # Gap difference is usually large, use log scale
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_reuse_gap_cdf.png", dpi=300)
    print(f"✓ Saved Plot 4: Reuse Gap CDF")
    plt.close()


def main():
    if not INPUT_FILE.exists():
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    sweep_results = data.get("sweep_results", [])
    adv_metrics = data.get("advanced_metrics", {})

    # 1. Print table
    if sweep_results:
        print_result_table(sweep_results)
    else:
        print("⚠ No sweep results found.")

    # 2. Generate plots
    if sweep_results:
        plot_1_hit_rate_trend(sweep_results)

    if adv_metrics:
        plot_2_shared_fraction_cdf(adv_metrics.get("shared_fractions", []))
        plot_3_block_hits_histogram(adv_metrics.get("block_hits", []))
        plot_4_reuse_gap_cdf(adv_metrics.get("reuse_gaps", []))

    print(f"\nAll plots saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()