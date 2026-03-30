"""
Robustness checks for the main regression analysis.

Tests:
1. Alternative volatility windows: [+1,+3], [+1,+10]
2. Lexicon-only uncertainty scores (no model)
3. Prepared remarks vs. Q&A section uncertainty
4. Subsample: large vs. small firms (by market cap median)
5. Subsample: by sector
6. Winsorization sensitivity

Reads: data/processed/panel_dataset.parquet
Outputs: output/tables/robustness_results.csv
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at given quantiles."""
    low_val = series.quantile(lower)
    high_val = series.quantile(upper)
    return series.clip(low_val, high_val)


def run_regression(df, dep_var, key_indep, controls, fe_vars, cluster_var=None):
    """Run a single OLS regression and return key coefficient info."""
    indep_vars = [key_indep] + controls
    subset = df.dropna(subset=[dep_var] + indep_vars)
    if len(subset) < 10:
        return None

    y = subset[dep_var]
    X = subset[indep_vars].copy()

    for fe_var in fe_vars:
        if fe_var in subset.columns:
            dummies = pd.get_dummies(subset[fe_var], prefix=fe_var, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X).astype(float)

    try:
        if cluster_var and cluster_var in subset.columns:
            model = sm.OLS(y, X).fit(
                cov_type="cluster", cov_kwds={"groups": subset[cluster_var]}
            )
        else:
            model = sm.OLS(y, X).fit(cov_type="HC1")
    except Exception as e:
        log.warning("Regression failed: %s", e)
        return None

    return {
        "coefficient": model.params.get(key_indep, np.nan),
        "std_error": model.bse.get(key_indep, np.nan),
        "t_stat": model.tvalues.get(key_indep, np.nan),
        "p_value": model.pvalues.get(key_indep, np.nan),
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    panel_path = os.path.join(project_root, "data", "processed", "panel_dataset.parquet")
    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(panel_path):
        log.error("Panel dataset not found. Run build_panel.py first.")
        sys.exit(1)

    panel = pd.read_parquet(panel_path)
    log.info("Panel: %d observations", len(panel))

    # Prepare variables
    panel["uncertainty_score_pct"] = panel["uncertainty_score"] * 100
    panel["uncertainty_share_pct"] = panel["uncertainty_share"] * 100
    panel["post_call_vol_bps"] = panel["post_call_vol"] * 10000
    panel["abnormal_vol_bps"] = panel["abnormal_vol"] * 10000

    if "post_vol_1_3" in panel.columns:
        panel["post_vol_1_3_bps"] = panel["post_vol_1_3"] * 10000
    if "post_vol_1_10" in panel.columns:
        panel["post_vol_1_10_bps"] = panel["post_vol_1_10"] * 10000

    if "prepared_uncertainty_score" in panel.columns:
        panel["prepared_score_pct"] = panel["prepared_uncertainty_score"] * 100
    if "qa_uncertainty_score" in panel.columns:
        panel["qa_score_pct"] = panel["qa_uncertainty_score"] * 100
    if "lexicon_uncertainty_score" in panel.columns:
        panel["lexicon_score_pct"] = panel["lexicon_uncertainty_score"] * 100

    # Winsorize
    for col in ["post_call_vol_bps", "abnormal_vol_bps", "uncertainty_score_pct",
                "uncertainty_share_pct", "pre_call_vol",
                "log_market_cap", "earnings_surprise"]:
        if col in panel.columns and panel[col].notna().sum() > 0:
            panel[col] = winsorize(panel[col].dropna()).reindex(panel.index)

    # Controls
    controls = ["pre_call_vol"]
    if panel["log_market_cap"].notna().sum() > len(panel) * 0.3:
        controls.append("log_market_cap")
    if panel["earnings_surprise"].notna().sum() > len(panel) * 0.3:
        controls.append("earnings_surprise")

    fe_vars = ["year_quarter", "sector"]
    cluster_var = "ticker"

    results = []

    # ===== Test 1: Alternative volatility windows =====
    log.info("\n=== Robustness 1: Alternative Volatility Windows ===")
    for dep_var, label in [
        ("post_call_vol_bps", "Vol [+1,+5]"),
        ("post_vol_1_3_bps", "Vol [+1,+3]"),
        ("post_vol_1_10_bps", "Vol [+1,+10]"),
        ("abnormal_vol_bps", "Abnormal Vol"),
    ]:
        if dep_var not in panel.columns:
            continue
        res = run_regression(
            panel, dep_var, "uncertainty_score_pct", controls, fe_vars, cluster_var
        )
        if res:
            res["test"] = "alt_windows"
            res["dep_var"] = label
            res["key_var"] = "Model Uncertainty"
            results.append(res)
            log.info(
                "%s: β=%.3f (SE=%.3f, p=%.4f, N=%d)",
                label, res["coefficient"], res["std_error"],
                res["p_value"], res["n_obs"],
            )

    # ===== Test 2: Lexicon-only uncertainty =====
    log.info("\n=== Robustness 2: Lexicon-Only Uncertainty ===")
    if "lexicon_score_pct" in panel.columns:
        res = run_regression(
            panel, "post_call_vol_bps", "lexicon_score_pct",
            controls, fe_vars, cluster_var,
        )
        if res:
            res["test"] = "lexicon_only"
            res["dep_var"] = "Vol [+1,+5]"
            res["key_var"] = "Lexicon Uncertainty"
            results.append(res)
            log.info(
                "Lexicon-only: β=%.3f (SE=%.3f, p=%.4f)",
                res["coefficient"], res["std_error"], res["p_value"],
            )

    # ===== Test 3: Prepared Remarks vs Q&A =====
    log.info("\n=== Robustness 3: Prepared Remarks vs Q&A ===")
    for var, label in [
        ("prepared_score_pct", "Prepared Remarks"),
        ("qa_score_pct", "Q&A Section"),
    ]:
        if var not in panel.columns:
            continue
        res = run_regression(
            panel, "post_call_vol_bps", var, controls, fe_vars, cluster_var
        )
        if res:
            res["test"] = "section_split"
            res["dep_var"] = "Vol [+1,+5]"
            res["key_var"] = label
            results.append(res)
            log.info(
                "%s: β=%.3f (SE=%.3f, p=%.4f)",
                label, res["coefficient"], res["std_error"], res["p_value"],
            )

    # ===== Test 4: Large vs Small firms =====
    log.info("\n=== Robustness 4: Large vs Small Firms ===")
    if panel["log_market_cap"].notna().sum() > 100:
        median_mcap = panel["log_market_cap"].median()
        for label, subset in [
            ("Large Firms", panel[panel["log_market_cap"] >= median_mcap]),
            ("Small Firms", panel[panel["log_market_cap"] < median_mcap]),
        ]:
            res = run_regression(
                subset, "post_call_vol_bps", "uncertainty_score_pct",
                controls, fe_vars, cluster_var,
            )
            if res:
                res["test"] = "size_split"
                res["dep_var"] = "Vol [+1,+5]"
                res["key_var"] = label
                results.append(res)
                log.info(
                    "%s: β=%.3f (SE=%.3f, p=%.4f, N=%d)",
                    label, res["coefficient"], res["std_error"],
                    res["p_value"], res["n_obs"],
                )

    # ===== Test 5: By sector (top 3 sectors) =====
    log.info("\n=== Robustness 5: By Sector ===")
    top_sectors = panel["sector"].value_counts().head(5).index
    for sector in top_sectors:
        subset = panel[panel["sector"] == sector]
        controls_no_sector = [c for c in controls]
        res = run_regression(
            subset, "post_call_vol_bps", "uncertainty_score_pct",
            controls_no_sector, ["year_quarter"], cluster_var,
        )
        if res:
            res["test"] = "sector_split"
            res["dep_var"] = "Vol [+1,+5]"
            res["key_var"] = sector
            results.append(res)
            log.info(
                "%s: β=%.3f (SE=%.3f, p=%.4f, N=%d)",
                sector, res["coefficient"], res["std_error"],
                res["p_value"], res["n_obs"],
            )

    # ===== Save all results =====
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_dir, "robustness_results.csv"), index=False
    )
    log.info("\nRobustness results saved to output/tables/robustness_results.csv")

    # Print summary table
    log.info("\n=== ROBUSTNESS SUMMARY ===")
    summary_cols = ["test", "key_var", "dep_var", "coefficient",
                    "std_error", "p_value", "n_obs"]
    if not results_df.empty:
        log.info("\n%s", results_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
