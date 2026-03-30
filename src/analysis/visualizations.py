"""
Produces all visualizations for the project report and presentation.

Figures:
  1. Distribution of uncertainty scores across transcripts
  2. Uncertainty scores over time (quarterly average)
  3. Scatter plot: uncertainty score vs. post-call volatility
  4. Sector-level average uncertainty bar chart
  5. Model performance: ROC curve
  6. Coefficient forest plot from robustness checks

Reads:
  data/processed/transcript_scores.parquet
  data/processed/panel_dataset.parquet
  output/tables/robustness_results.csv

Outputs:
  output/figures/*.png
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("muted")
FIG_DIR = None


def save(fig, name, dpi=200):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", path)


# ------------------------------------------------------------------ #
# Figure 1 – Distribution of uncertainty scores
# ------------------------------------------------------------------ #
def plot_uncertainty_distribution(scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(scores["uncertainty_score"], bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Model Uncertainty Score")
    ax.set_ylabel("Number of Transcripts")
    ax.set_title("Distribution of Uncertainty Scores")

    ax = axes[1]
    ax.hist(scores["uncertainty_share"], bins=60, color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Share of Uncertain Sentences")
    ax.set_ylabel("Number of Transcripts")
    ax.set_title("Distribution of Uncertainty Share")

    fig.tight_layout()
    save(fig, "uncertainty_distribution.png")


# ------------------------------------------------------------------ #
# Figure 2 – Uncertainty over time
# ------------------------------------------------------------------ #
def plot_uncertainty_over_time(scores):
    df = scores.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    df["year_quarter"] = df["event_date"].dt.to_period("Q")

    quarterly = df.groupby("year_quarter").agg(
        mean_score=("uncertainty_score", "mean"),
        median_score=("uncertainty_score", "median"),
        count=("uncertainty_score", "size"),
    ).reset_index()
    quarterly["year_quarter_str"] = quarterly["year_quarter"].astype(str)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(
        quarterly["year_quarter_str"],
        quarterly["mean_score"],
        marker="o", linewidth=2, color=PALETTE[0], label="Mean"
    )
    ax1.plot(
        quarterly["year_quarter_str"],
        quarterly["median_score"],
        marker="s", linewidth=2, color=PALETTE[2], linestyle="--", label="Median"
    )
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Uncertainty Score")
    ax1.set_title("Average Management Uncertainty Over Time")
    ax1.legend(loc="upper left")

    # Rotate labels
    step = max(1, len(quarterly) // 12)
    ax1.set_xticks(range(0, len(quarterly), step))
    ax1.set_xticklabels(
        quarterly["year_quarter_str"].iloc[::step], rotation=45, ha="right"
    )

    ax2 = ax1.twinx()
    ax2.bar(
        quarterly["year_quarter_str"],
        quarterly["count"],
        alpha=0.15, color="grey", width=0.8,
    )
    ax2.set_ylabel("Number of Calls", color="grey")

    fig.tight_layout()
    save(fig, "uncertainty_over_time.png")


# ------------------------------------------------------------------ #
# Figure 3 – Scatter: uncertainty vs. post-call volatility
# ------------------------------------------------------------------ #
def plot_scatter_uncertainty_vol(panel):
    df = panel.dropna(subset=["uncertainty_score", "post_call_vol"]).copy()
    if len(df) < 10:
        log.warning("Not enough data for scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df["uncertainty_score"] * 100,
        df["post_call_vol"] * 10000,
        alpha=0.15, s=8, color=PALETTE[0],
    )

    # Binned means
    df["score_bin"] = pd.qcut(df["uncertainty_score"], 20, duplicates="drop")
    binned = df.groupby("score_bin").agg(
        x=("uncertainty_score", "mean"),
        y=("post_call_vol", "mean"),
    ).reset_index()
    ax.plot(binned["x"] * 100, binned["y"] * 10000, "o-", color=PALETTE[3], linewidth=2, markersize=5)

    ax.set_xlabel("Uncertainty Score (%)")
    ax.set_ylabel("Post-Call Volatility (bps)")
    ax.set_title("Uncertainty vs. Post-Call Volatility")
    fig.tight_layout()
    save(fig, "scatter_uncertainty_vol.png")


# ------------------------------------------------------------------ #
# Figure 4 – Sector bar chart
# ------------------------------------------------------------------ #
def plot_sector_uncertainty(scores):
    if "sector" not in scores.columns:
        log.warning("No sector column – skipping sector chart.")
        return

    sector_avg = (
        scores.groupby("sector")["uncertainty_score"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(sector_avg) * 0.45)))
    sector_avg.plot.barh(ax=ax, color=PALETTE[0])
    ax.set_xlabel("Mean Uncertainty Score")
    ax.set_title("Average Management Uncertainty by Sector")
    fig.tight_layout()
    save(fig, "sector_uncertainty.png")


# ------------------------------------------------------------------ #
# Figure 5 – Robustness coefficient forest plot
# ------------------------------------------------------------------ #
def plot_robustness_forest(project_root):
    rob_path = os.path.join(project_root, "output", "tables", "robustness_results.csv")
    if not os.path.exists(rob_path):
        log.warning("Robustness results not found – skipping forest plot.")
        return

    rob = pd.read_csv(rob_path)
    if rob.empty:
        return

    rob = rob.sort_values("coefficient")
    rob["label"] = rob["test"] + " | " + rob["key_var"]

    fig, ax = plt.subplots(figsize=(8, max(4, len(rob) * 0.4)))
    y_pos = range(len(rob))
    ax.errorbar(
        rob["coefficient"], y_pos,
        xerr=1.96 * rob["std_error"],
        fmt="o", color=PALETTE[0], capsize=3,
    )
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(rob["label"])
    ax.set_xlabel("Coefficient on Uncertainty")
    ax.set_title("Robustness Checks – Coefficient Estimates (95% CI)")
    fig.tight_layout()
    save(fig, "robustness_forest.png")


# ------------------------------------------------------------------ #
# Figure 6 – Correlation heat map of key variables
# ------------------------------------------------------------------ #
def plot_correlation_heatmap(panel):
    key_cols = [
        "uncertainty_score", "uncertainty_share",
        "post_call_vol", "pre_call_vol", "abnormal_vol",
        "log_market_cap", "earnings_surprise",
    ]
    cols = [c for c in key_cols if c in panel.columns and panel[c].notna().sum() > 10]
    if len(cols) < 3:
        log.warning("Not enough variable overlap for heatmap.")
        return

    corr = panel[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Matrix of Key Variables")
    fig.tight_layout()
    save(fig, "correlation_heatmap.png")


# ================================================================== #
# Main
# ================================================================== #
def main():
    global FIG_DIR

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    FIG_DIR = os.path.join(project_root, "output", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load data
    scores_path = os.path.join(project_root, "data", "processed", "transcript_scores.parquet")
    panel_path = os.path.join(project_root, "data", "processed", "panel_dataset.parquet")

    if not os.path.exists(scores_path):
        log.error("transcript_scores.parquet not found. Run predict_uncertainty.py first.")
        sys.exit(1)

    scores = pd.read_parquet(scores_path)
    log.info("Loaded %d transcript scores.", len(scores))

    # Figures from scores only
    plot_uncertainty_distribution(scores)
    plot_uncertainty_over_time(scores)
    plot_sector_uncertainty(scores)

    # Figures needing panel data
    if os.path.exists(panel_path):
        panel = pd.read_parquet(panel_path)
        log.info("Loaded %d panel observations.", len(panel))
        plot_scatter_uncertainty_vol(panel)
        plot_correlation_heatmap(panel)
    else:
        log.warning("Panel dataset not found – skipping panel-dependent figures.")

    # Robustness forest
    plot_robustness_forest(project_root)

    log.info("All visualizations complete.")


if __name__ == "__main__":
    main()
