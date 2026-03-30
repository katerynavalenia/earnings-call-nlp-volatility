"""
Run the main panel regression: does management uncertainty predict
post-call stock volatility?

Model:
  PostCallVol_it = β1·UncertaintyScore_it + β2·PreCallVol_it
                 + β3·log(MarketCap)_it + β4·EarningsSurprise_it
                 + Sector_FE + YearQuarter_FE + ε_it

Reads: data/processed/panel_dataset.parquet
Outputs: output/tables/regression_results.csv, printed summary
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize a series at the given quantiles."""
    low_val = series.quantile(lower)
    high_val = series.quantile(upper)
    return series.clip(low_val, high_val)


def compute_vif(X):
    """Compute Variance Inflation Factors for a design matrix."""
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data


def run_ols_with_fe(panel, dep_var, indep_vars, fe_vars, cluster_var=None):
    """
    Run OLS regression with fixed effects (absorbed as dummies).
    Returns the fitted model.
    """
    df = panel.dropna(subset=[dep_var] + indep_vars)

    y = df[dep_var]
    X = df[indep_vars].copy()

    for fe_var in fe_vars:
        if fe_var in df.columns:
            dummies = pd.get_dummies(df[fe_var], prefix=fe_var, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    X = X.astype(float)

    if cluster_var and cluster_var in df.columns:
        # OLS with clustered standard errors
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cluster_var]},
        )
    else:
        # Robust standard errors
        model = sm.OLS(y, X).fit(cov_type="HC1")

    return model, X[indep_vars + ["const"]]


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
    log.info("Panel: %d observations, %d tickers", len(panel), panel["ticker"].nunique())

    # Winsorize continuous variables at 1%/99%
    for col in ["post_call_vol", "pre_call_vol", "abnormal_vol",
                "uncertainty_score", "uncertainty_share",
                "log_market_cap", "earnings_surprise"]:
        if col in panel.columns and panel[col].notna().sum() > 0:
            panel[col] = winsorize(panel[col].dropna()).reindex(panel.index)

    # Scale uncertainty score to percentage points for interpretability
    # (already 0-1 scale, multiply by 100 so coefficients are per 1pp)
    panel["uncertainty_score_pct"] = panel["uncertainty_score"] * 100
    panel["uncertainty_share_pct"] = panel["uncertainty_share"] * 100

    # Convert to basis points for interpretable coefficients
    panel["post_call_vol_bps"] = panel["post_call_vol"] * 10000
    panel["abnormal_vol_bps"] = panel["abnormal_vol"] * 10000

    # ===== Model 1: Baseline (uncertainty only) =====
    log.info("\n" + "=" * 60)
    log.info("MODEL 1: Baseline — Uncertainty → Post-Call Volatility")
    log.info("=" * 60)

    indep_base = ["uncertainty_score_pct"]
    model1, _ = run_ols_with_fe(
        panel, "post_call_vol_bps", indep_base,
        fe_vars=["year_quarter"],
        cluster_var="ticker",
    )
    log.info("\n%s", model1.summary().tables[1].as_text())

    # ===== Model 2: With pre-call volatility control =====
    log.info("\n" + "=" * 60)
    log.info("MODEL 2: + Pre-Call Volatility Control")
    log.info("=" * 60)

    indep_controls = ["uncertainty_score_pct", "pre_call_vol"]
    # Only add controls that have data
    if panel["log_market_cap"].notna().sum() > len(panel) * 0.3:
        indep_controls.append("log_market_cap")
    if panel["earnings_surprise"].notna().sum() > len(panel) * 0.3:
        indep_controls.append("earnings_surprise")

    model2, X_for_vif = run_ols_with_fe(
        panel, "post_call_vol_bps", indep_controls,
        fe_vars=["year_quarter", "sector"],
        cluster_var="ticker",
    )
    log.info("\n%s", model2.summary().tables[1].as_text())

    # ===== Model 3: Uncertainty share (alternative measure) =====
    log.info("\n" + "=" * 60)
    log.info("MODEL 3: Uncertainty Share (alternative measure)")
    log.info("=" * 60)

    indep_share = ["uncertainty_share_pct"] + [
        c for c in indep_controls if c != "uncertainty_score_pct"
    ]
    model3, _ = run_ols_with_fe(
        panel, "post_call_vol_bps", indep_share,
        fe_vars=["year_quarter", "sector"],
        cluster_var="ticker",
    )
    log.info("\n%s", model3.summary().tables[1].as_text())

    # ===== Model 4: Abnormal volatility as dependent variable =====
    log.info("\n" + "=" * 60)
    log.info("MODEL 4: Abnormal Volatility as DV")
    log.info("=" * 60)

    model4, _ = run_ols_with_fe(
        panel, "abnormal_vol_bps", ["uncertainty_score_pct"],
        fe_vars=["year_quarter", "sector"],
        cluster_var="ticker",
    )
    log.info("\n%s", model4.summary().tables[1].as_text())

    # ===== VIF Check =====
    log.info("\n" + "=" * 60)
    log.info("VIF CHECK")
    log.info("=" * 60)
    try:
        vif = compute_vif(X_for_vif.dropna())
        log.info("\n%s", vif.to_string(index=False))
    except Exception as e:
        log.warning("Could not compute VIF: %s", e)

    # ===== Save results =====
    results = []
    for name, model in [("baseline", model1), ("full_controls", model2),
                         ("share_measure", model3), ("abnormal_vol", model4)]:
        for param, coef in model.params.items():
            if param.startswith("year_quarter") or param.startswith("sector"):
                continue
            results.append({
                "model": name,
                "variable": param,
                "coefficient": coef,
                "std_error": model.bse.get(param, np.nan),
                "t_stat": model.tvalues.get(param, np.nan),
                "p_value": model.pvalues.get(param, np.nan),
                "r_squared": model.rsquared,
                "n_obs": int(model.nobs),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_dir, "regression_results.csv"), index=False
    )
    log.info("Regression results saved to output/tables/regression_results.csv")

    # ===== Interpretable summary =====
    log.info("\n" + "=" * 60)
    log.info("INTERPRETATION (Model 2 — Full Controls)")
    log.info("=" * 60)
    beta = model2.params.get("uncertainty_score_pct", np.nan)
    se = model2.bse.get("uncertainty_score_pct", np.nan)
    pval = model2.pvalues.get("uncertainty_score_pct", np.nan)
    log.info(
        "A 1 percentage point increase in the model-predicted uncertainty score "
        "is associated with a %.2f basis point change in post-call realized "
        "volatility (SE=%.2f, p=%.4f).",
        beta, se, pval,
    )
    log.info(
        "Equivalently, a 10pp increase in uncertainty → %.1f bps change in "
        "post-call vol.",
        beta * 10,
    )


if __name__ == "__main__":
    main()
