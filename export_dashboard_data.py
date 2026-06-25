# export_dashboard_data.py
# Purpose: Export all pre-computed yield curve analytics from a Colab/pandas
# environment into structured JSON files consumed by the React dashboard.
# Run this script after all model fitting is complete. Output goes to ./dashboard-data/

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("./dashboard-data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: safe JSON serialiser (handles numpy scalars, NaN → null)
# ─────────────────────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)

def write_json(filename: str, payload: dict):
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(payload, f, cls=NumpyEncoder, indent=2)
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ {filename:40s}  {size_kb:7.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT 1 — yield_surface.json
# Inputs expected:
#   yield_df  : pd.DataFrame — rows = dates (DatetimeIndex),
#               columns = maturity labels e.g. ["1M","3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"]
#               values  = yield in % (annualised)
# ─────────────────────────────────────────────────────────────────────────────
def export_yield_surface(yield_df: pd.DataFrame):
    print("\n[1/4] Exporting 3D yield surface …")
    assert isinstance(yield_df.index, pd.DatetimeIndex), "yield_df must have a DatetimeIndex"
    assert yield_df.shape[1] >= 2, "yield_df needs at least two maturity columns"

    dates = [d.strftime("%Y-%m-%d") for d in yield_df.index]
    maturities = list(yield_df.columns)
    z_matrix = yield_df.values.tolist()          # list[list[float|None]]

    payload = {
        "dates": dates,
        "maturities": maturities,
        "z": z_matrix,
        "meta": {
            "description": "U.S. Treasury constant-maturity yields (% p.a.)",
            "n_dates": len(dates),
            "n_maturities": len(maturities),
            "date_range": [dates[0], dates[-1]],
        },
    }
    write_json("yield_surface.json", payload)
    print(f"     Dates: {len(dates)} | Maturities: {maturities}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT 2 — ns_calibration.json
# Inputs expected:
#   ns_params_df : pd.DataFrame — DatetimeIndex, columns = ["beta1","beta2","beta3","lambda","rmse"]
#                  (add "beta4","lambda2" columns if NSS model is fitted)
#   observed_df  : pd.DataFrame — same structure as yield_df (observed yields per date)
#   fitted_df    : pd.DataFrame — same structure, Nelson-Siegel fitted values
#   fitted_nss_df: pd.DataFrame or None — NSS fitted values (optional)
# ─────────────────────────────────────────────────────────────────────────────
def export_ns_calibration(
    ns_params_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    fitted_df: pd.DataFrame,
    fitted_nss_df: pd.DataFrame = None,
):
    print("\n[2/4] Exporting Nelson-Siegel calibration …")

    params_records = []
    for date, row in ns_params_df.iterrows():
        rec = {"date": date.strftime("%Y-%m-%d")}
        rec.update(row.to_dict())
        params_records.append(rec)

    cross_sections = {}
    for date in observed_df.index:
        d = date.strftime("%Y-%m-%d")
        obs = observed_df.loc[date].to_dict()
        fit = fitted_df.loc[date].to_dict() if date in fitted_df.index else {}
        nss = (
            fitted_nss_df.loc[date].to_dict()
            if fitted_nss_df is not None and date in fitted_nss_df.index
            else {}
        )
        cross_sections[d] = {"observed": obs, "fitted_ns": fit, "fitted_nss": nss}

    payload = {
        "params_history": params_records,
        "cross_sections": cross_sections,
        "maturities": list(observed_df.columns),
        "meta": {
            "description": "Nelson-Siegel (and optionally Svensson) calibration results",
            "has_nss": fitted_nss_df is not None,
            "n_dates": len(params_records),
        },
    }
    write_json("ns_calibration.json", payload)
    print(f"     Dates with cross-sections: {len(cross_sections)}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT 3 — kalman_factors.json
# Inputs expected:
#   ols_factors_df    : pd.DataFrame — DatetimeIndex, columns = ["beta1","beta2","beta3"]
#                        OLS/static NS factors extracted date-by-date
#   kalman_factors_df : pd.DataFrame — same columns, Kalman-filtered state estimates
# ─────────────────────────────────────────────────────────────────────────────
def export_kalman_factors(
    ols_factors_df: pd.DataFrame,
    kalman_factors_df: pd.DataFrame,
):
    print("\n[3/4] Exporting Kalman filter factor series …")
    assert set(["beta1", "beta2", "beta3"]).issubset(ols_factors_df.columns), \
        "ols_factors_df must contain beta1, beta2, beta3"

    def df_to_series(df):
        return {
            col: {
                "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                "values": df[col].tolist(),
            }
            for col in ["beta1", "beta2", "beta3"]
        }

    payload = {
        "ols_factors": df_to_series(ols_factors_df),
        "kalman_factors": df_to_series(kalman_factors_df),
        "factor_labels": {
            "beta1": "Level (β₁)",
            "beta2": "Slope (β₂)",
            "beta3": "Curvature (β₃)",
        },
        "meta": {
            "description": "OLS cross-sectional vs. Kalman-smoothed Dynamic NS factors",
            "n_dates": len(ols_factors_df),
        },
    }
    write_json("kalman_factors.json", payload)
    print(f"     Factor observations: {len(ols_factors_df)}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT 4 — recession_probability.json
# Inputs expected:
#   recession_prob_df : pd.DataFrame — DatetimeIndex, column "prob" = P(recession in 12m) ∈ [0,1]
#   nber_recessions   : list[dict]   — each dict has {"start":"YYYY-MM-DD","end":"YYYY-MM-DD","label":"..."}
#   model_metrics     : dict         — e.g. {"pseudo_r2": 0.42, "auc_roc": 0.88, "brier_score": 0.09}
# ─────────────────────────────────────────────────────────────────────────────
def export_recession_probability(
    recession_prob_df: pd.DataFrame,
    nber_recessions: list,
    model_metrics: dict = None,
):
    print("\n[4/4] Exporting recession probability series …")
    assert "prob" in recession_prob_df.columns, "recession_prob_df must have a 'prob' column"

    latest_prob = float(recession_prob_df["prob"].dropna().iloc[-1])
    latest_date = recession_prob_df["prob"].dropna().index[-1].strftime("%Y-%m-%d")

    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in recession_prob_df.index],
        "probabilities": recession_prob_df["prob"].tolist(),
        "current": {
            "date": latest_date,
            "probability": round(latest_prob * 100, 2),   # convert to %
        },
        "nber_recessions": nber_recessions,
        "model_metrics": model_metrics or {},
        "meta": {
            "description": "12-month-ahead recession probability from Probit model",
            "n_observations": len(recession_prob_df),
        },
    }
    write_json("recession_probability.json", payload)
    print(f"     Latest estimate: {latest_prob*100:.1f}% as of {latest_date}")
    print(f"     NBER recession windows: {len(nber_recessions)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — call each exporter with your actual DataFrames
# Replace the SAMPLE DATA blocks below with your real variables
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Dashboard Data Export")
    print("  Output directory:", OUTPUT_DIR.resolve())
    print("=" * 60)

    # ── REPLACE BELOW WITH YOUR REAL DATAFRAMES ──────────────────────────────

    # ── 1. Yield surface ──────────────────────────────────────────────────────
    # Example: yield_df is already in your Colab environment
    # export_yield_surface(yield_df)

    # ── 2. NS calibration ─────────────────────────────────────────────────────
    # export_ns_calibration(ns_params_df, observed_df, fitted_df, fitted_nss_df=None)

    # ── 3. Kalman factors ─────────────────────────────────────────────────────
    # export_kalman_factors(ols_factors_df, kalman_factors_df)

    # ── 4. Recession probability ──────────────────────────────────────────────
    # nber = [
    #     {"start": "2001-03-01", "end": "2001-11-30", "label": "2001 Dot-com"},
    #     {"start": "2007-12-01", "end": "2009-06-30", "label": "2008 GFC"},
    #     {"start": "2020-02-01", "end": "2020-04-30", "label": "2020 COVID"},
    # ]
    # metrics = {"pseudo_r2": 0.42, "auc_roc": 0.88, "brier_score": 0.09}
    # export_recession_probability(recession_prob_df, nber, metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # SAMPLE DATA (generates realistic placeholder JSON so the dashboard loads)
    # Delete this block once you plug in real DataFrames above
    # ─────────────────────────────────────────────────────────────────────────
    dates = pd.date_range("1993-01-01", "2024-12-31", freq="ME")
    maturities = ["1M","3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"]
    mat_years  = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])

    rng = np.random.default_rng(42)
    beta1 = 5 + rng.normal(0, 0.8, len(dates)).cumsum() * 0.04
    beta2 = -2 + rng.normal(0, 0.5, len(dates)).cumsum() * 0.03
    beta3 = 1.5 + rng.normal(0, 0.4, len(dates)).cumsum() * 0.02
    lam = 0.0609

    def ns_curve(b1, b2, b3, l, tau):
        f = (1 - np.exp(-l * tau)) / (l * tau)
        return b1 + b2 * f + b3 * (f - np.exp(-l * tau))

    z = np.array([[ns_curve(beta1[i], beta2[i], beta3[i], lam, t) + rng.normal(0, 0.05)
                   for t in mat_years] for i in range(len(dates))])
    yield_df = pd.DataFrame(z, index=dates, columns=maturities)

    ns_params = pd.DataFrame({
        "beta1": beta1, "beta2": beta2, "beta3": beta3,
        "lambda": lam, "rmse": rng.uniform(0.01, 0.08, len(dates))
    }, index=dates)
    fitted_z = np.array([[ns_curve(beta1[i], beta2[i], beta3[i], lam, t)
                          for t in mat_years] for i in range(len(dates))])
    fitted_df = pd.DataFrame(fitted_z, index=dates, columns=maturities)
    observed_df = yield_df

    kalman_b1 = pd.Series(beta1, index=dates).rolling(3).mean().bfill()
    kalman_b2 = pd.Series(beta2, index=dates).rolling(3).mean().bfill()
    kalman_b3 = pd.Series(beta3, index=dates).rolling(3).mean().bfill()
    ols_factors_df    = pd.DataFrame({"beta1": beta1, "beta2": beta2, "beta3": beta3}, index=dates)
    kalman_factors_df = pd.DataFrame({"beta1": kalman_b1, "beta2": kalman_b2, "beta3": kalman_b3}, index=dates)

    spread = yield_df["10Y"] - yield_df["2Y"]
    logit  = -0.5 + -1.8 * spread / spread.std()
    prob   = 1 / (1 + np.exp(-logit))
    recession_prob_df = pd.DataFrame({"prob": prob.values}, index=dates)
    nber_recessions = [
        {"start": "2001-03-01", "end": "2001-11-30", "label": "2001 Dot-com"},
        {"start": "2007-12-01", "end": "2009-06-30", "label": "2008 GFC"},
        {"start": "2020-02-01", "end": "2020-04-30", "label": "2020 COVID"},
    ]
    model_metrics = {"pseudo_r2": 0.42, "auc_roc": 0.88, "brier_score": 0.09}

    # Run all exports
    export_yield_surface(yield_df)
    export_ns_calibration(ns_params, observed_df, fitted_df)
    export_kalman_factors(ols_factors_df, kalman_factors_df)
    export_recession_probability(recession_prob_df, nber_recessions, model_metrics)

    print("\n" + "=" * 60)
    print("  Export complete. Copy /dashboard-data/ into")
    print("  your React project's /public/ folder.")
    print("=" * 60)