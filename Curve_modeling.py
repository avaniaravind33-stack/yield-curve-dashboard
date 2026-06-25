# ============================================================
# PROJECT 6: YIELD CURVE MODELING, NELSON-SIEGEL & RECESSION PREDICTION
# Google Colab Implementation
# Author: Avani Aravind | Date: 2026
# Category: Fixed Income / Macroeconomics | Difficulty: Advanced
# ============================================================
#
# SYSTEM ARCHITECTURE:
#
#   [FRED API / Treasury.gov]
#          │
#          ▼
#   [Data Pipeline]  ──► Treasury Yields (30Y daily)
#          │             NBER Recession Dates
#          │             Macro indicators (CPI, Unemployment)
#          ▼
#   [Preprocessing & Feature Engineering]
#          │  ── Clean, interpolate, align dates
#          │  ── Compute spreads, curvature metrics
#          ▼
#   ┌──────────────────────────────────┐
#   │  Nelson-Siegel / NSS Calibration │  scipy.optimize per date
#   │  β1=Level, β2=Slope, β3=Curvature│
#   └──────────────┬───────────────────┘
#                  │
#          ▼
#   [Dynamic Nelson-Siegel + Kalman Filter]
#          │  ── State-space: βt = A*βt-1 + noise
#          │  ── Kalman update per observation
#          ▼
#   [Probit Recession Model]
#          │  ── Features: spreads, DNS factors
#          │  ── Binary: P(recession in 12M)
#          ▼
#   [Visualizations & Dashboard]
#          ── 3D yield surface, factor plots,
#          ── recession probability, forward curve
#
# ============================================================


# ════════════════════════════════════════════════════════════
# CELL 1: INSTALLATION & ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════

"""
Run this cell first in Google Colab to install all dependencies.
Restart runtime after installation if prompted.
"""

# !pip install pandas-datareader fredapi scipy numpy matplotlib plotly \
#              statsmodels scikit-learn filterpy tqdm requests --quiet

# For Colab: mount Google Drive (optional, for saving outputs)
# from google.colab import drive
# drive.mount('/content/drive')


# ════════════════════════════════════════════════════════════
# CELL 2: IMPORTS & GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════

import os
import warnings
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from plotly import graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    import pandas_datareader.data as web
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False
    print("pandas-datareader not available; will use fallback data.")

try:
    from fredapi import Fred
    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False
    print("fredapi not available; will use pandas-datareader FRED endpoint.")

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("filterpy not available; using custom Kalman implementation.")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Global Plot Style ─────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#ffffff",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "lines.linewidth":   1.8,
})

ACCENT_COLORS = {
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "red":    "#f85149",
    "orange": "#d29922",
    "purple": "#bc8cff",
    "teal":   "#39d353",
    "gray":   "#8b949e",
}

pio.templates.default = "plotly_dark"

# ── Constants ─────────────────────────────────────────────
FRED_API_KEY = "c4b586a97670988886851abf9e25be93"   # ← replace or set env var
START_DATE   = "1993-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")

# Standard Treasury maturities in years
MATURITIES   = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
MAT_LABELS   = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y",
                 "10Y", "20Y", "30Y"]

# FRED series IDs for each maturity
FRED_SERIES = {
    "1M":  "DGS1MO",
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "3Y":  "DGS3",
    "5Y":  "DGS5",
    "7Y":  "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

# FRED macro series
MACRO_SERIES = {
    "RECESSION": "USREC",       # NBER recession indicator (monthly)
    "UNRATE":    "UNRATE",      # Unemployment rate
    "CPIAUCSL":  "CPIAUCSL",   # CPI (all urban consumers)
    "FEDFUNDS":  "FEDFUNDS",   # Federal funds rate
}

print("✅ Configuration complete.")
print(f"   Data range : {START_DATE} → {END_DATE}")
print(f"   Maturities : {MAT_LABELS}")


# ════════════════════════════════════════════════════════════
# CELL 3: DATA COLLECTION PIPELINE
# ════════════════════════════════════════════════════════════

class TreasuryDataPipeline:
    """
    Pulls U.S. Treasury yield data and macro indicators from FRED.

    Supports both fredapi (preferred, needs API key) and
    pandas-datareader as a fallback.  All data is aligned to a
    common daily business-day index and forward/backward filled
    for minor gaps (holidays, occasional missing prints).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self._fred   = None
        self._init_client()

    # ── Initialise FRED client ──────────────────────────────
    def _init_client(self):
        if FREDAPI_AVAILABLE and self.api_key and self.api_key != "YOUR_FRED_API_KEY_HERE":
            self._fred = Fred(api_key=self.api_key)
            logger.info("Using fredapi client.")
        elif DATAREADER_AVAILABLE:
            logger.info("Using pandas-datareader FRED endpoint.")
        else:
            logger.warning("No FRED client available; will generate synthetic data.")

    # ── Low-level fetch (single series) ────────────────────
    def _fetch_series(self, series_id: str,
                      start: str, end: str) -> pd.Series:
        try:
            if self._fred is not None:
                raw = self._fred.get_series(series_id,
                                            observation_start=start,
                                            observation_end=end)
                raw.name = series_id
                return raw
            elif DATAREADER_AVAILABLE:
                raw = web.DataReader(series_id, "fred", start, end)
                return raw[series_id]
        except Exception as exc:
            logger.warning(f"Could not fetch {series_id}: {exc}")
        return pd.Series(dtype=float, name=series_id)

    # ── Pull all Treasury yields ────────────────────────────
    def fetch_yields(self, start: str = START_DATE,
                     end: str = END_DATE) -> pd.DataFrame:
        """
        Returns a DataFrame with columns = maturity labels (1M … 30Y)
        and a daily DatetimeIndex.  Values are annualised yield in %.
        """
        logger.info("Fetching Treasury yield data from FRED …")
        frames = {}
        for label, series_id in FRED_SERIES.items():
            s = self._fetch_series(series_id, start, end)
            if not s.empty:
                frames[label] = s
            else:
                logger.warning(f"  Series {series_id} returned empty.")

        if not frames:
            logger.warning("No yield data fetched — generating synthetic data.")
            return self._synthetic_yields(start, end)

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.reindex(
            pd.date_range(df.index.min(), df.index.max(), freq="B")
        )
        # Fill short gaps (weekends already dropped; ≤3-day gaps acceptable)
        df = df.ffill(limit=3).bfill(limit=3)
        df = df.dropna(how="all")

        # Reorder columns by maturity
        ordered = [c for c in MAT_LABELS if c in df.columns]
        df = df[ordered]

        logger.info(f"  ✓ Yields: {df.shape[0]} rows × {df.shape[1]} maturities")
        return df

    # ── Pull macro series ───────────────────────────────────
    def fetch_macro(self, start: str = START_DATE,
                    end: str = END_DATE) -> pd.DataFrame:
        """
        Returns a DataFrame of macro indicators aligned to a monthly index.
        """
        logger.info("Fetching macro data from FRED …")
        frames = {}
        for label, series_id in MACRO_SERIES.items():
            s = self._fetch_series(series_id, start, end)
            if not s.empty:
                frames[label] = s

        if not frames:
            logger.warning("No macro data — generating synthetic recession flags.")
            return self._synthetic_macro(start, end)

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        logger.info(f"  ✓ Macro: {df.shape[0]} rows × {df.shape[1]} indicators")
        return df

    # ── Synthetic fallback data (demo mode) ─────────────────
    @staticmethod
    def _synthetic_yields(start: str, end: str) -> pd.DataFrame:
        """
        Generates plausible synthetic yield curves when FRED is unavailable.
        Uses a Nelson-Siegel process with realistic random-walk betas.
        """
        logger.info("Generating SYNTHETIC yield data (demo mode).")
        dates = pd.date_range(start, end, freq="B")
        n     = len(dates)
        tau   = np.array(MATURITIES)

        np.random.seed(42)
        # Random-walk latent factors (mean-reverting)
        b1 = np.cumsum(np.random.normal(0, 0.01, n)) + 4.0   # level  ~4%
        b2 = np.cumsum(np.random.normal(0, 0.008, n)) - 1.5  # slope  ~ -1.5%
        b3 = np.cumsum(np.random.normal(0, 0.005, n)) + 0.5  # curv   ~0.5%

        # Mean-reversion clamp
        b1 = np.clip(b1, 0.5, 8.0)
        b2 = np.clip(b2, -4.0, 2.0)
        b3 = np.clip(b3, -3.0, 3.0)
        lam = 0.0609

        yields = []
        for i in range(n):
            load1 = 1.0
            load2 = (1 - np.exp(-tau / lam)) / (tau / lam)
            load3 = load2 - np.exp(-tau / lam)
            y = b1[i] * load1 + b2[i] * load2 + b3[i] * load3
            y += np.random.normal(0, 0.04, len(tau))
            yields.append(np.clip(y, 0.01, 15.0))

        df = pd.DataFrame(yields, index=dates, columns=MAT_LABELS)
        return df

    @staticmethod
    def _synthetic_macro(start: str, end: str) -> pd.DataFrame:
        """Synthetic monthly macro data for demo mode."""
        dates = pd.date_range(start, end, freq="MS")
        n     = len(dates)
        np.random.seed(99)
        rec   = np.zeros(n, dtype=int)
        # Insert a few fake recession episodes
        for s, e in [(36, 56), (90, 98), (180, 196), (240, 258), (324, 340)]:
            if e < n:
                rec[s:e] = 1
        df = pd.DataFrame({
            "RECESSION": rec,
            "UNRATE":    np.clip(4.5 + np.cumsum(np.random.normal(0, 0.05, n)), 3, 12),
            "CPIAUCSL":  np.cumprod(1 + np.random.normal(0.003, 0.002, n)) * 100,
            "FEDFUNDS":  np.clip(4.0 + np.cumsum(np.random.normal(0, 0.05, n)), 0, 20),
        }, index=dates)
        return df


# ── Run pipeline ───────────────────────────────────────────
pipeline  = TreasuryDataPipeline(api_key=FRED_API_KEY)
yields_df = pipeline.fetch_yields(start=START_DATE, end=END_DATE)
macro_df  = pipeline.fetch_macro(start=START_DATE, end=END_DATE)

print(f"\n📊 Yields DataFrame  : {yields_df.shape}")
print(yields_df.tail(3).to_string())
print(f"\n📊 Macro DataFrame   : {macro_df.shape}")
print(macro_df.tail(3).to_string())


# ════════════════════════════════════════════════════════════
# CELL 4: DATA CLEANING & FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════

class YieldCurveFeatureEngineer:
    """
    Computes standard fixed-income features from a yield DataFrame:

    Spreads
    ───────
    • 2Y10Y   — classic recession predictor
    • 3M10Y   — Fed-preferred recession indicator
    • 2Y30Y   — long-end steepness
    • 5Y30Y   — curve butterfly wing

    Curvature & shape metrics
    ─────────────────────────
    • butterfly   2×5Y − (2Y + 10Y)
    • convexity   rough second derivative at 5Y

    Rolling stats
    ─────────────
    • 20-day z-scores of key spreads
    • 12-month rolling means/stdevs
    """

    REQUIRED_COLS = ["2Y", "5Y", "10Y"]

    def __init__(self, yields: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLS if c not in yields.columns]
        if missing:
            raise ValueError(f"Missing maturity columns: {missing}")
        self.yields = yields.copy()

    def compute(self) -> pd.DataFrame:
        df = self.yields.copy()

        # ── Spreads (basis points = × 100) ─────────────────
        def spread_bp(long_col, short_col, name):
            if long_col in df and short_col in df:
                df[name] = (df[long_col] - df[short_col]) * 100

        spread_bp("10Y", "2Y",  "spread_2Y10Y_bp")
        spread_bp("10Y", "3M",  "spread_3M10Y_bp")
        spread_bp("30Y", "2Y",  "spread_2Y30Y_bp")
        spread_bp("30Y", "5Y",  "spread_5Y30Y_bp")
        spread_bp("10Y", "1Y",  "spread_1Y10Y_bp")
        spread_bp("5Y",  "2Y",  "spread_2Y5Y_bp")

        # ── Curve inversion flag ────────────────────────────
        if "spread_2Y10Y_bp" in df:
            df["inverted_2Y10Y"] = (df["spread_2Y10Y_bp"] < 0).astype(int)
        if "spread_3M10Y_bp" in df:
            df["inverted_3M10Y"] = (df["spread_3M10Y_bp"] < 0).astype(int)

        # ── Butterfly (curvature) ───────────────────────────
        df["butterfly"] = 2 * df["5Y"] - (df["2Y"] + df["10Y"])

        # ── Level, slope, curvature (naïve PCA proxies) ────
        if "30Y" in df and "1M" in df:
            df["curve_level"] = df[["2Y", "5Y", "10Y", "30Y"]].mean(axis=1)
            df["curve_slope"] = df["30Y"] - df["1M"]

        # ── Rolling z-scores (252 trading days ≈ 1 year) ───
        for col in ["spread_2Y10Y_bp", "spread_3M10Y_bp", "butterfly"]:
            if col in df:
                roll    = df[col].rolling(252, min_periods=60)
                df[f"{col}_zscore"] = (df[col] - roll.mean()) / roll.std()

        # ── 20-day & 60-day changes in key rates ───────────
        for col in ["2Y", "10Y"]:
            if col in df:
                df[f"{col}_chg20d"] = df[col].diff(20)
                df[f"{col}_chg60d"] = df[col].diff(60)

        return df

    @staticmethod
    def align_with_macro(yields_feat: pd.DataFrame,
                         macro: pd.DataFrame) -> pd.DataFrame:
        """
        Left-joins monthly macro data onto the daily yield feature frame.
        Recession flag and macro variables are forward-filled within month.
        """
        macro_daily = macro.reindex(yields_feat.index, method="ffill")
        combined    = yields_feat.join(macro_daily, how="left")
        return combined


# ── Run feature engineering ────────────────────────────────
fe       = YieldCurveFeatureEngineer(yields_df)
feat_df  = fe.compute()
full_df  = YieldCurveFeatureEngineer.align_with_macro(feat_df, macro_df)

print("✅ Feature engineering complete.")
print(f"   Shape: {full_df.shape}")
print("\nNew features:")
new_cols = [c for c in full_df.columns if c not in yields_df.columns]
for c in new_cols:
    print(f"  {c}")
# In Curve modeling.py, within the feature engineering section:
# Calculate the 12-month change in UNRATE
# Assuming 'full_df' is the DataFrame being used for feature engineering
full_df['UNRATE_chg12m'] = full_df['UNRATE'].diff(12)

# Handle NaN values introduced by the diff operation
# Depending on your modeling approach, you might drop rows with NaNs or fill them appropriately.
# For a simple fix, dropping them might be a starting point.
full_df = full_df.dropna(subset=['UNRATE_chg12m'])


# ════════════════════════════════════════════════════════════
# CELL 5: NELSON-SIEGEL MODEL
# ════════════════════════════════════════════════════════════

class NelsonSiegelModel:
    """
    Static Nelson-Siegel yield curve model.

    The NS formula (Diebold & Li 2006):

        y(τ) = β1
             + β2 · [(1 - e^{-τ/λ}) / (τ/λ)]
             + β3 · [(1 - e^{-τ/λ}) / (τ/λ) - e^{-τ/λ}]

    Parameters
    ──────────
    β1  : Level factor (long-end yield)
    β2  : Slope factor (short - long spread, ≈ negative of 2-10 spread)
    β3  : Curvature / hump factor
    λ   : Decay / shape parameter (controls where hump occurs)
          Typical value: 0.0609 → hump at ~30 months

    Calibration
    ───────────
    Minimise sum-of-squared residuals between fitted and observed yields
    using scipy.optimize.minimize (L-BFGS-B) per date.
    """

    # Default initial guess and bounds
    BETA_INIT  = [5.0, -2.0, 1.0]
    LAMBDA_INIT = 0.0609
    BOUNDS = [
        (0.01,  20.0),   # β1 ≥ 0 (long rate > 0)
        (-15.0, 15.0),   # β2
        (-15.0, 15.0),   # β3
        (0.005, 3.0),    # λ
    ]

    def __init__(self, maturities: List[float] = MATURITIES,
                 fit_lambda: bool = False,
                 fixed_lambda: float = 0.0609):
        self.tau        = np.array(maturities)
        self.fit_lambda = fit_lambda
        self.lam        = fixed_lambda

    # ── NS loadings ─────────────────────────────────────────
    def _loadings(self, lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = self.tau
        L1  = np.ones(len(tau))
        exp_term = np.exp(-tau / lam)
        L2  = (1 - exp_term) / (tau / lam)
        L3  = L2 - exp_term
        return L1, L2, L3

    # ── Curve fit ───────────────────────────────────────────
    def predict(self, betas: np.ndarray,
                lam: Optional[float] = None) -> np.ndarray:
        lam    = lam or self.lam
        L1, L2, L3 = self._loadings(lam)
        return betas[0] * L1 + betas[1] * L2 + betas[2] * L3

    # ── Calibrate to a single observation ───────────────────
    def fit_single(self, observed: np.ndarray,
                   tau_mask: Optional[np.ndarray] = None
                   ) -> Dict:
        """
        Fit NS model to one cross-section of yields.

        Parameters
        ──────────
        observed : array of yields (same length as self.tau)
        tau_mask : boolean mask for available maturities

        Returns dict with keys: betas, lambda, fitted, rmse, success
        """
        mask = tau_mask if tau_mask is not None else ~np.isnan(observed)
        obs  = observed[mask]
        tau  = self.tau[mask]

        if len(obs) < 3:
            return {"success": False, "betas": np.full(3, np.nan),
                    "lambda": self.lam, "fitted": np.full(len(self.tau), np.nan),
                    "rmse": np.nan}

        if self.fit_lambda:
            x0 = self.BETA_INIT + [self.LAMBDA_INIT]
            bounds = self.BOUNDS

            def objective(x):
                b, lam = x[:3], x[3]
                exp_t  = np.exp(-tau / lam)
                L2     = (1 - exp_t) / (tau / lam)
                L3     = L2 - exp_t
                fitted = b[0] + b[1] * L2 + b[2] * L3
                return np.sum((fitted - obs) ** 2)

            res = minimize(objective, x0, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 500, "ftol": 1e-12})
            betas, lam_fit = res.x[:3], res.x[3]
        else:
            lam_fit = self.lam
            exp_t   = np.exp(-tau / lam_fit)
            L2      = (1 - exp_t) / (tau / lam_fit)
            L3      = L2 - exp_t
            X       = np.column_stack([np.ones(len(tau)), L2, L3])
            result  = np.linalg.lstsq(X, obs, rcond=None)
            betas   = result[0]
            res     = type("R", (), {"success": True})()

        # Full-maturity fitted curve
        exp_all = np.exp(-self.tau / lam_fit)
        L2_all  = (1 - exp_all) / (self.tau / lam_fit)
        L3_all  = L2_all - exp_all
        fitted  = betas[0] + betas[1] * L2_all + betas[2] * L3_all

        resid = obs - (betas[0] + betas[1] *
                       (1 - np.exp(-tau / lam_fit)) / (tau / lam_fit) +
                       betas[2] * ((1 - np.exp(-tau / lam_fit)) / (tau / lam_fit)
                                   - np.exp(-tau / lam_fit)))
        rmse = np.sqrt(np.mean(resid ** 2))

        return {"success": getattr(res, "success", True),
                "betas":   betas,
                "lambda":  lam_fit,
                "fitted":  fitted,
                "rmse":    rmse}

    # ── Calibrate across all dates ───────────────────────────
    def fit_time_series(self, yields: pd.DataFrame,
                        maturity_cols: List[str] = MAT_LABELS,
                        verbose: bool = True) -> pd.DataFrame:
        """
        Fits NS model for every date in yields DataFrame.

        Returns a DataFrame with columns:
          beta1, beta2, beta3, lambda, rmse
        indexed by date.
        """
        cols  = [c for c in maturity_cols if c in yields.columns]
        tau   = np.array([MATURITIES[MAT_LABELS.index(c)] for c in cols])
        model = NelsonSiegelModel(maturities=tau.tolist(),
                                  fit_lambda=self.fit_lambda,
                                  fixed_lambda=self.lam)

        results = []
        for i, (date, row) in enumerate(yields[cols].iterrows()):
            obs  = row.values.astype(float)
            mask = ~np.isnan(obs)
            if mask.sum() < 3:
                results.append({"date": date, "beta1": np.nan,
                                 "beta2": np.nan, "beta3": np.nan,
                                 "lambda": np.nan, "rmse": np.nan})
                continue
            res = model.fit_single(obs, tau_mask=mask)
            results.append({"date": date,
                             "beta1":  res["betas"][0],
                             "beta2":  res["betas"][1],
                             "beta3":  res["betas"][2],
                             "lambda": res["lambda"],
                             "rmse":   res["rmse"]})
            if verbose and i % 500 == 0:
                logger.info(f"  NS calibration: {i}/{len(yields)} dates")

        df = pd.DataFrame(results).set_index("date")
        return df


# ── Calibrate NS model across full history ─────────────────
print("⚙️  Calibrating Nelson-Siegel model (this may take a moment)…")
ns_model   = NelsonSiegelModel(fit_lambda=False, fixed_lambda=0.0609)
# Use monthly sampling for speed during development; switch to daily for production
monthly_yields = yields_df.resample("MS").first()
ns_factors     = ns_model.fit_time_series(monthly_yields, verbose=True)

print(f"\n✅ NS calibration complete. Shape: {ns_factors.shape}")
print(ns_factors.tail(5).to_string())
print(f"\nMedian RMSE: {ns_factors['rmse'].median():.4f}%")


# ════════════════════════════════════════════════════════════
# CELL 6: NELSON-SIEGEL-SVENSSON (NSS) MODEL
# ════════════════════════════════════════════════════════════

class NelsonSiegelSvenssonModel:
    """
    Extended Nelson-Siegel-Svensson model (Svensson 1994).

    Adds a second hump term for better long-end fit:

        y(τ) = β1
             + β2 · f(τ, λ1)
             + β3 · g(τ, λ1)
             + β4 · g(τ, λ2)

    where
        f(τ, λ) = (1 - e^{-τ/λ}) / (τ/λ)
        g(τ, λ) = f(τ, λ) - e^{-τ/λ}

    Parameters
    ──────────
    β1  : Level
    β2  : Slope
    β3  : First curvature (hump)
    β4  : Second curvature (long-end hump)
    λ1  : First decay parameter (~0.06)
    λ2  : Second decay parameter (~0.5)

    Used by the ECB and many central banks for official curve estimation.
    """

    BOUNDS = [
        (-2.0, 20.0),   # β1
        (-20.0, 20.0),  # β2
        (-20.0, 20.0),  # β3
        (-20.0, 20.0),  # β4
        (0.005, 5.0),   # λ1
        (0.005, 10.0),  # λ2
    ]
    X0 = [4.0, -1.0, 0.5, 0.5, 0.0609, 0.5]

    def __init__(self, maturities: List[float] = MATURITIES):
        self.tau = np.array(maturities)

    @staticmethod
    def _f(tau: np.ndarray, lam: float) -> np.ndarray:
        return (1 - np.exp(-tau / lam)) / (tau / lam)

    @staticmethod
    def _g(tau: np.ndarray, lam: float) -> np.ndarray:
        return NelsonSiegelSvenssonModel._f(tau, lam) - np.exp(-tau / lam)

    def predict(self, params: np.ndarray) -> np.ndarray:
        b1, b2, b3, b4, l1, l2 = params
        tau = self.tau
        return (b1
                + b2 * self._f(tau, l1)
                + b3 * self._g(tau, l1)
                + b4 * self._g(tau, l2))

    def fit_single(self, observed: np.ndarray) -> Dict:
        """Fit NSS to one cross-section using differential evolution."""
        mask = ~np.isnan(observed)
        obs  = observed[mask]
        tau  = self.tau[mask]

        def objective(params):
            b1, b2, b3, b4, l1, l2 = params
            y_hat = (b1
                     + b2 * self._f(tau, l1)
                     + b3 * self._g(tau, l1)
                     + b4 * self._g(tau, l2))
            return np.sum((y_hat - obs) ** 2)

        # Two-stage: global search → local refinement
        try:
            de_res = differential_evolution(objective, self.BOUNDS,
                                            seed=42, maxiter=300,
                                            tol=1e-8, workers=1)
            res    = minimize(objective, de_res.x, method="L-BFGS-B",
                              bounds=self.BOUNDS,
                              options={"maxiter": 500})
            params  = res.x
            success = res.success
        except Exception:
            params  = np.array(self.X0)
            success = False

        fitted = self.predict(params)
        resid  = obs - self.predict(params)[mask]
        rmse   = np.sqrt(np.mean(resid ** 2))

        return {"params": params, "fitted": fitted,
                "rmse": rmse, "success": success}

    def fit_sample_dates(self, yields: pd.DataFrame,
                         n_dates: int = 12) -> Dict:
        """
        Fit NSS on a representative set of dates (spread across history).
        Full daily calibration is expensive; sample monthly/quarterly.
        """
        cols  = [c for c in MAT_LABELS if c in yields.columns]
        dates = yields.index[::max(1, len(yields) // n_dates)]
        results = {}
        for date in dates:
            obs = yields.loc[date, cols].values.astype(float)
            res = self.fit_single(obs)
            results[date] = res
            logger.info(f"  NSS @ {date.date()} | RMSE={res['rmse']:.4f}%")
        return results


# ── Fit NSS on a handful of key dates ──────────────────────
print("⚙️  Calibrating Nelson-Siegel-Svensson on sample dates…")
nss_model  = NelsonSiegelSvenssonModel(maturities=MATURITIES)
nss_sample = nss_model.fit_sample_dates(monthly_yields, n_dates=12)
print(f"✅ NSS done. Sample dates fitted: {len(nss_sample)}")


# ════════════════════════════════════════════════════════════
# CELL 7: DYNAMIC NELSON-SIEGEL + KALMAN FILTER
# ════════════════════════════════════════════════════════════

class DynamicNelsonSiegelKalman:
    """
    Dynamic Nelson-Siegel State-Space Model (Diebold, Rudebusch, Aruoba 2006).

    State equation (VAR(1) on factors):
        βt = A · βt-1 + ηt,   ηt ~ N(0, Q)

    Observation equation:
        yt = Λ(λ) · βt + εt,  εt ~ N(0, H)

    where
        yt  = (K × 1) vector of yields at maturities τ1..τK
        βt  = (3 × 1) [β1, β2, β3] latent factor vector
        Λ   = (K × 3) loading matrix (NS loadings)
        A   = (3 × 3) VAR(1) transition matrix
        Q   = (3 × 3) state noise covariance
        H   = (K × K) observation noise covariance (diagonal)

    Kalman Filter steps
    ───────────────────
    Predict:
        β_pred  = A · β_{t-1}
        P_pred  = A · P_{t-1} · A' + Q

    Update (available maturities only):
        v       = y_obs - Λ_obs · β_pred          (innovation)
        S       = Λ_obs · P_pred · Λ_obs' + H_obs (innovation cov)
        K       = P_pred · Λ_obs' · inv(S)         (Kalman gain)
        β_t     = β_pred + K · v
        P_t     = (I - K · Λ_obs) · P_pred
    """

    def __init__(self, maturities: List[float],
                 lambda_ns: float = 0.0609):
        self.tau  = np.array(maturities)
        self.lam  = lambda_ns
        self.K    = len(maturities)   # number of yields
        self.n    = 3                 # number of latent factors

        # Build static loading matrix Λ (K × 3)
        self.Lambda = self._build_loadings(self.tau, lambda_ns)

    @staticmethod
    def _build_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
        exp_t = np.exp(-tau / lam)
        L1    = np.ones(len(tau))
        L2    = (1 - exp_t) / (tau / lam)
        L3    = L2 - exp_t
        return np.column_stack([L1, L2, L3])   # (K × 3)

    def _init_params(self, beta_series: pd.DataFrame) -> Tuple:
        """
        Initialise A, Q, H from OLS fit of VAR(1) on pre-estimated betas.
        """
        B  = beta_series[["beta1", "beta2", "beta3"]].dropna().values  # (T, 3)
        T  = len(B)

        # OLS: Bt = A · B_{t-1}
        Y  = B[1:].T    # (3, T-1)
        X  = B[:-1].T   # (3, T-1)
        A  = Y @ X.T @ np.linalg.inv(X @ X.T)  # (3, 3)

        # State noise Q from residuals
        resid = Y - A @ X
        Q     = (resid @ resid.T) / (T - 1)

        # Observation noise H (diagonal, small)
        H = np.eye(self.K) * 0.01

        # Initial state
        b0 = B[0]
        P0 = np.eye(3) * 1.0

        return A, Q, H, b0, P0

    def filter(self, yields: pd.DataFrame,
               beta_init: pd.DataFrame,
               maturity_cols: List[str] = MAT_LABELS
               ) -> pd.DataFrame:
        """
        Run Kalman filter over the yield time series.

        Parameters
        ──────────
        yields      : DataFrame of observed yields (T × K)
        beta_init   : pre-estimated NS betas (for parameter initialisation)
        maturity_cols: ordered list matching self.tau

        Returns
        ───────
        DataFrame with smoothed factor estimates:
            beta1_kf, beta2_kf, beta3_kf  (Kalman-filtered)
            beta1_ns, beta2_ns, beta3_ns  (original OLS/NS for comparison)
        """
        cols = [c for c in maturity_cols if c in yields.columns]
        Y    = yields[cols].values.astype(float)   # (T, K)
        T    = len(Y)

        A, Q, H, b0, P0 = self._init_params(beta_init)

        # Storage
        beta_filt = np.zeros((T, 3))
        P_filt    = np.zeros((T, 3, 3))

        b = b0.copy()
        P = P0.copy()

        for t in range(T):
            # ── Predict ─────────────────────────────────────
            b_pred = A @ b
            P_pred = A @ P @ A.T + Q

            # ── Observed maturities at this date ────────────
            obs_mask = ~np.isnan(Y[t])
            if obs_mask.sum() >= 3:
                Lam_obs = self.Lambda[obs_mask]   # (k_obs, 3)
                H_obs   = H[np.ix_(obs_mask, obs_mask)]
                y_obs   = Y[t, obs_mask]

                # Innovation
                v = y_obs - Lam_obs @ b_pred
                S = Lam_obs @ P_pred @ Lam_obs.T + H_obs
                K = P_pred @ Lam_obs.T @ np.linalg.inv(S)

                # Update
                b = b_pred + K @ v
                P = (np.eye(3) - K @ Lam_obs) @ P_pred
            else:
                b = b_pred
                P = P_pred

            beta_filt[t] = b
            P_filt[t]    = P

        idx = yields.index
        kf_df = pd.DataFrame({
            "beta1_kf": beta_filt[:, 0],
            "beta2_kf": beta_filt[:, 1],
            "beta3_kf": beta_filt[:, 2],
        }, index=idx)

        # Merge with original NS estimates for comparison
        kf_df = kf_df.join(
            beta_init[["beta1", "beta2", "beta3"]].rename(
                columns={"beta1": "beta1_ns",
                         "beta2": "beta2_ns",
                         "beta3": "beta3_ns"}
            ), how="left"
        )
        return kf_df


# ── Run Dynamic NS + Kalman Filter ─────────────────────────
print("⚙️  Running Dynamic Nelson-Siegel Kalman Filter…")
dns_model  = DynamicNelsonSiegekFilter = DynamicNelsonSiegelKalman(
    maturities=[MATURITIES[MAT_LABELS.index(c)]
                for c in MAT_LABELS
                if c in monthly_yields.columns],
    lambda_ns=0.0609
)
kf_results = DynamicNelsonSiegekFilter.filter(
    monthly_yields, ns_factors
)
print(f"✅ Kalman filter complete. Shape: {kf_results.shape}")
print(kf_results.tail(5).to_string())


# ════════════════════════════════════════════════════════════
# CELL 8: FORWARD RATE EXTRACTION
# ════════════════════════════════════════════════════════════

class ForwardRateCalculator:
    """
    Extracts instantaneous and discrete forward rates from NS/NSS curves.

    Instantaneous forward rate (from NS):
        f(τ) = β1 + β2·e^{-τ/λ} + β3·(τ/λ)·e^{-τ/λ}

    Discrete forward rate between τ1 and τ2:
        f(τ1, τ2) = [τ2·y(τ2) - τ1·y(τ1)] / (τ2 - τ1)

    The forward curve reveals market expectations for future short rates,
    risk premium structure, and the pricing of duration risk.
    """

    @staticmethod
    def ns_instantaneous_forward(tau_grid: np.ndarray,
                                  betas: np.ndarray,
                                  lam: float = 0.0609) -> np.ndarray:
        """
        Compute the NS instantaneous forward rate curve.
        df(τ)/dτ is derived analytically from the NS yield curve.
        """
        b1, b2, b3 = betas[:3]
        et  = np.exp(-tau_grid / lam)
        return b1 + b2 * et + b3 * (tau_grid / lam) * et

    @staticmethod
    def discrete_forward(yields_row: pd.Series,
                          maturity_map: Dict[str, float]
                          ) -> pd.Series:
        """
        Compute 1Y × 1Y forward rates from observed yields.
        e.g., 1Y1Y, 2Y1Y, 5Y1Y, 9Y1Y (forward starting 1Y bonds).
        """
        mats  = sorted(maturity_map.items(), key=lambda x: x[1])
        names = [m[0] for m in mats]
        taus  = [m[1] for m in mats]
        ys    = yields_row.reindex(names).values.astype(float)

        forwards = {}
        for i in range(len(taus) - 1):
            t1, t2 = taus[i], taus[i + 1]
            y1, y2 = ys[i], ys[i + 1]
            if np.isnan(y1) or np.isnan(y2):
                continue
            fwd_name = f"{names[i]}×{names[i+1]}_fwd"
            forwards[fwd_name] = (t2 * y2 - t1 * y1) / (t2 - t1)

        return pd.Series(forwards)

    @staticmethod
    def compute_dv01(betas: np.ndarray, tau_grid: np.ndarray,
                     lam: float = 0.0609,
                     face_value: float = 1_000_000) -> pd.Series:
        """
        DV01 (Dollar Value of 01) — sensitivity of bond price to 1bp shift.
        Approximate via duration: DV01 ≈ Price × Modified Duration / 10000
        Uses simplified flat yield assumption per maturity.
        """
        b1, b2, b3 = betas[:3]
        et  = np.exp(-tau_grid / lam)
        L2  = (1 - et) / (tau_grid / lam)
        L3  = L2 - et
        yields_pct = b1 + b2 * L2 + b3 * L3   # %

        # Approximate modified duration ≈ τ (for zero-coupon bonds)
        mod_dur = tau_grid

        # DV01 ≈ Face × Duration / (1 + y) × 0.0001
        y_dec = yields_pct / 100
        dv01  = face_value * mod_dur / (1 + y_dec) * 0.0001

        return pd.Series(dv01,
                         index=[f"{t:.1f}Y" for t in tau_grid],
                         name="DV01_USD")


# ── Compute forward curves for key historical dates ─────────
mat_map = {c: MATURITIES[MAT_LABELS.index(c)]
           for c in MAT_LABELS if c in yields_df.columns}

key_dates = pd.to_datetime([
    "2000-01-01",  # Dot-com bubble
    "2006-06-01",  # Pre-GFC (flat/inverted)
    "2008-12-01",  # GFC trough
    "2013-06-01",  # Taper tantrum
    "2019-08-01",  # Pre-COVID (inverted)
    "2022-10-01",  # Post-hike cycle peak
])
key_dates = [d for d in key_dates if d in yields_df.index or
             yields_df.index.asof(d) is not pd.NaT]

frc = ForwardRateCalculator()
tau_fine = np.linspace(0.08, 30, 300)

forward_curves = {}
for d in key_dates:
    # Snap to nearest available date
    nearest = yields_df.index[yields_df.index.get_indexer([d], method="nearest")[0]]
    row     = yields_df.loc[nearest]
    res     = ns_model.fit_single(row.values.astype(float))
    if res["success"]:
        fwd = frc.ns_instantaneous_forward(tau_fine, res["betas"])
        forward_curves[nearest] = fwd

print(f"✅ Forward curves computed for {len(forward_curves)} key dates.")


# ════════════════════════════════════════════════════════════
# CELL 9: PROBIT RECESSION MODEL
# ════════════════════════════════════════════════════════════

class RecessionProbitModel:
    """
    Probit regression to predict P(recession in next 12 months).

    Model specification
    ───────────────────
        P(Rec_{t+12} = 1 | Xt) = Φ(α + β'Xt)

    where
        Φ  = standard normal CDF
        Xt = vector of yield-curve features at time t

    Features used
    ─────────────
    • 2Y10Y spread (basis points)      — primary predictor
    • 3M10Y spread (basis points)      — Fed-preferred indicator
    • Butterfly curvature
    • 20-day z-score of 2Y10Y spread
    • Level of 10Y yield
    • Monthly change in 2Y yield
    • Unemployment rate (if available)

    Evaluation
    ──────────
    • Pseudo R² (McFadden)
    • ROC-AUC
    • Brier score
    • Time-series cross-validation (5 folds)
    • Comparison to NBER recession dates
    """

    FEATURE_COLS = [
        "spread_2Y10Y_bp",
        "spread_3M10Y_bp",
        "butterfly",
        "spread_2Y10Y_bp_zscore",
        "10Y",
        "2Y_chg60d",
    ]

    def __init__(self, horizon_months: int = 12):
        self.horizon = horizon_months
        self.model   = None
        self.scaler  = StandardScaler()
        self.feature_cols = self.FEATURE_COLS.copy()

    def _prepare_data(self, full_df: pd.DataFrame
                       ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix X and forward-shifted recession label y.

        y[t] = 1 if ANY month in [t, t + horizon] is a recession.
        """
        # Resample to monthly (end of month)
        monthly = full_df.resample("ME").last()

        # Forward-shifted recession label
        if "RECESSION" in monthly.columns:
            rec = monthly["RECESSION"].fillna(0)
        else:
            # Create a synthetic label from curve inversion
            rec = (monthly.get("inverted_2Y10Y", pd.Series(0, index=monthly.index))
                   .fillna(0))

        # Rolling OR over next 12 months → 1 if any recession within horizon
        rec_future = rec.rolling(self.horizon, min_periods=1).max().shift(-self.horizon)

        # Align features
        avail_feats = [c for c in self.feature_cols if c in monthly.columns]
        X = monthly[avail_feats].copy()

        # Add unemployment if available
        if "UNRATE" in monthly.columns:
            X["UNRATE"] = monthly["UNRATE"]
            X["UNRATE_chg12m"] = monthly["UNRATE"].diff(12)

        # Drop rows with NaN in either X or y
        combined = X.join(rec_future.rename("y")).dropna()
        X_clean  = combined.drop("y", axis=1)
        y_clean  = combined["y"].astype(int)
        self.feature_cols_used = X_clean.columns.tolist()

        return X_clean, y_clean

    def fit(self, full_df: pd.DataFrame) -> "RecessionProbitModel":
        """Fit the probit model and print diagnostics."""
        X, y = self._prepare_data(full_df)

        # Scale features
        X_sc = self.scaler.fit_transform(X)
        X_sm  = sm.add_constant(X_sc)

        # statsmodels Probit
        self.model = sm.Probit(y, X_sm)
        try:
            self.result = self.model.fit(method="newton",
                                          maxiter=200,
                                          disp=False)
        except Exception:
            self.result = self.model.fit(method="bfgs",
                                          maxiter=500,
                                          disp=False)

        self._X = X
        self._y = y
        self._X_sc = X_sc

        print("✅ Probit model fitted.")
        print(self.result.summary2())
        return self

    def predict_proba(self, full_df: pd.DataFrame) -> pd.Series:
        """Return P(recession) for all dates in full_df."""
        monthly = full_df.resample("ME").last()
        avail   = [c for c in self.feature_cols_used if c in monthly.columns]
        X       = monthly[avail].dropna()
        X_sc    = self.scaler.transform(X)
        X_sm    = sm.add_constant(X_sc, has_constant="add")
        proba   = self.result.predict(X_sm)
        return pd.Series(proba, index=X.index, name="recession_prob")

    def evaluate(self) -> Dict:
        """Compute in-sample evaluation metrics."""
        X_sm  = sm.add_constant(self._X_sc, has_constant="add")
        preds = self.result.predict(X_sm)
        y     = self._y.values

        auc       = roc_auc_score(y, preds)
        brier     = np.mean((preds - y) ** 2)
        pseudo_r2 = self.result.prsquared

        print(f"\n📊 Model Evaluation")
        print(f"   ROC-AUC   : {auc:.4f}")
        print(f"   Brier     : {brier:.4f}")
        print(f"   McFadden R²: {pseudo_r2:.4f}")

        # Classification at 0.5 threshold
        pred_bin = (preds >= 0.5).astype(int)
        print(f"\n{classification_report(y, pred_bin, target_names=['Expansion','Recession'])}")

        return {"auc": auc, "brier": brier, "pseudo_r2": pseudo_r2}

    def time_series_cv(self, n_splits: int = 5) -> pd.DataFrame:
        """
        Walk-forward cross-validation with TimeSeriesSplit.
        Avoids look-ahead bias: always train on past, test on future.
        """
        tscv    = TimeSeriesSplit(n_splits=n_splits)
        X_arr   = self._X_sc
        y_arr   = self._y.values
        records = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

            X_sm_tr = sm.add_constant(X_tr)
            X_sm_te = sm.add_constant(X_te, has_constant="add")

            try:
                m  = sm.Probit(y_tr, X_sm_tr).fit(disp=False, maxiter=200)
                pr = m.predict(X_sm_te)
                if len(np.unique(y_te)) > 1:
                    auc = roc_auc_score(y_te, pr)
                else:
                    auc = np.nan
                brier = np.mean((pr - y_te) ** 2)
                records.append({"fold": fold + 1, "auc": auc, "brier": brier})
            except Exception as exc:
                records.append({"fold": fold + 1, "auc": np.nan,
                                 "brier": np.nan})

        cv_df = pd.DataFrame(records)
        print("\n📊 Time-Series Cross-Validation Results:")
        print(cv_df.to_string(index=False))
        print(f"\n   Mean AUC  : {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
        print(f"   Mean Brier: {cv_df['brier'].mean():.4f} ± {cv_df['brier'].std():.4f}")
        return cv_df


# ── Fit probit model ────────────────────────────────────────
print("⚙️  Fitting Probit Recession Model…")
probit = RecessionProbitModel(horizon_months=12)
probit.fit(full_df)
metrics   = probit.evaluate()
cv_scores = probit.time_series_cv()
rec_proba = probit.predict_proba(full_df)

print(f"\n✅ Recession probabilities computed. Shape: {rec_proba.shape}")


# ════════════════════════════════════════════════════════════
# CELL 10: VISUALISATIONS
# ════════════════════════════════════════════════════════════

class YieldCurveDashboard:
    """
    Production-quality visualisation suite for the yield curve project.

    Charts
    ──────
    1. NS Factor Time Series (Level / Slope / Curvature)
    2. Model Fit: NS Fitted vs. Observed (selected dates)
    3. Recession Probability with NBER Shading
    4. 2Y10Y Spread + Recession Shading (historical)
    5. 3D Yield Surface (Plotly, interactive)
    6. Animated Yield Curve (12-month rolling, Plotly)
    7. Forward Rate Curves at Key Dates
    8. ROC Curve (probit model)
    """

    def __init__(self, yields: pd.DataFrame,
                 ns_factors: pd.DataFrame,
                 kf_results: pd.DataFrame,
                 full_df: pd.DataFrame,
                 rec_proba: pd.Series,
                 ns_model: NelsonSiegelModel):
        self.yields     = yields
        self.ns         = ns_factors
        self.kf         = kf_results
        self.full       = full_df
        self.rec_proba  = rec_proba
        self.ns_model   = ns_model

    # ── Helper: shade NBER recessions ───────────────────────
    def _shade_recessions(self, ax):
        if "RECESSION" not in self.full.columns:
            return
        rec = self.full["RECESSION"].resample("ME").last().ffill()
        in_rec = False
        start  = None
        for date, val in rec.items():
            if val == 1 and not in_rec:
                start  = date
                in_rec = True
            elif val == 0 and in_rec:
                ax.axvspan(start, date, color="#f85149",
                           alpha=0.12, zorder=0)
                in_rec = False
        if in_rec and start is not None:
            ax.axvspan(start, rec.index[-1],
                       color="#f85149", alpha=0.12, zorder=0)

    # ── Chart 1: NS Factor Time Series ──────────────────────
    def plot_ns_factors(self):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Dynamic Nelson-Siegel Latent Factors\n"
                     "Level · Slope · Curvature",
                     fontsize=15, fontweight="bold", y=1.01)

        factor_meta = [
            ("beta1_ns", "beta1_kf", "Level (β₁)", ACCENT_COLORS["blue"],
             "Long-run yield level → parallel curve shifts"),
            ("beta2_ns", "beta2_kf", "Slope (β₂)", ACCENT_COLORS["orange"],
             "Short-end minus long-end → steepness / recession signal"),
            ("beta3_ns", "beta3_kf", "Curvature (β₃)", ACCENT_COLORS["purple"],
             "Mid-maturity hump → butterfly risk"),
        ]

        for ax, (ns_col, kf_col, title, color, desc) in zip(axes, factor_meta):
            if ns_col in self.kf.columns:
                ax.plot(self.kf.index, self.kf[ns_col],
                        color=color, alpha=0.45,
                        label="OLS / NS", linewidth=1.2)
            if kf_col in self.kf.columns:
                ax.plot(self.kf.index, self.kf[kf_col],
                        color=color, alpha=0.95,
                        label="Kalman Filter", linewidth=1.8)
            ax.axhline(0, color="#30363d", linewidth=0.8, linestyle="--")
            self._shade_recessions(ax)
            ax.set_ylabel(title, color=color, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(desc, fontsize=9,
                         color=ACCENT_COLORS["gray"], pad=3)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("ns_factors.png",dpi=150, bbox_inches='tight', facecolor='white') # keep your other kwargs here)
        plt.show()
        print("  Saved: ns_factors.png")

    # ── Chart 2: NS Fit vs. Observed ────────────────────────
    def plot_model_fit(self, n_dates: int = 6):
        sample_idx = np.linspace(0, len(self.yields) - 1, n_dates, dtype=int)
        dates      = self.yields.index[sample_idx]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle("Nelson-Siegel Fitted vs. Observed Yields",
                     fontsize=14, fontweight="bold")
        axes = axes.flatten()
        tau_fine = np.linspace(0.08, 30, 200)

        cols = [c for c in MAT_LABELS if c in self.yields.columns]
        tau  = [MATURITIES[MAT_LABELS.index(c)] for c in cols]

        for ax, date in zip(axes, dates):
            obs = self.yields.loc[date, cols].values.astype(float)
            res = self.ns_model.fit_single(obs)

            ax.scatter(tau, obs, color=ACCENT_COLORS["teal"],
                       s=55, zorder=5, label="Observed")
            if res["success"]:
                exp_f = np.exp(-tau_fine / self.ns_model.lam)
                L2_f  = (1 - exp_f) / (tau_fine / self.ns_model.lam)
                L3_f  = L2_f - exp_f
                fit   = (res["betas"][0]
                         + res["betas"][1] * L2_f
                         + res["betas"][2] * L3_f)
                ax.plot(tau_fine, fit,
                        color=ACCENT_COLORS["orange"],
                        linewidth=2.0, label=f"NS fit (RMSE={res['rmse']:.3f}%)")

            ax.set_title(date.strftime("%d %b %Y"),
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Maturity (years)", fontsize=8)
            ax.set_ylabel("Yield (%)", fontsize=8)
            ax.legend(fontsize=7)
            ax.set_xlim(0, 31)

        plt.tight_layout()
        plt.savefig("ns_model_fit.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print("  Saved: ns_model_fit.png")

    # ── Chart 3: Recession Probability ──────────────────────
    def plot_recession_probability(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                         sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle("Recession Probability Model (Probit)\n"
                     "12-Month Ahead Prediction vs. NBER Recessions",
                     fontsize=14, fontweight="bold")

        # Probability panel
        ax1.fill_between(self.rec_proba.index,
                          self.rec_proba.values,
                          alpha=0.4, color=ACCENT_COLORS["red"])
        ax1.plot(self.rec_proba.index, self.rec_proba.values,
                 color=ACCENT_COLORS["red"], linewidth=1.8)
        ax1.axhline(0.5, color=ACCENT_COLORS["orange"],
                    linewidth=1.2, linestyle="--",
                    label="50% threshold")
        self._shade_recessions(ax1)
        ax1.set_ylabel("P(Recession in next 12M)", fontweight="bold")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper left")

        # Spread panel
        if "spread_2Y10Y_bp" in self.full.columns:
            spread = self.full["spread_2Y10Y_bp"].resample("ME").last()
            ax2.fill_between(spread.index, spread.values, 0,
                              where=spread.values < 0,
                              color=ACCENT_COLORS["red"], alpha=0.5,
                              label="Inverted (< 0 bps)")
            ax2.fill_between(spread.index, spread.values, 0,
                              where=spread.values >= 0,
                              color=ACCENT_COLORS["blue"], alpha=0.3,
                              label="Normal (> 0 bps)")
            ax2.plot(spread.index, spread.values,
                     color=ACCENT_COLORS["blue"], linewidth=1.2)
            ax2.axhline(0, color=ACCENT_COLORS["gray"],
                        linewidth=0.8, linestyle="--")
            self._shade_recessions(ax2)
            ax2.set_ylabel("2Y10Y Spread (bps)", fontweight="bold")
            ax2.legend(loc="upper left", fontsize=8)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator(4))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("recession_probability.png", dpi=300, bbox_inches='tight'),
        plt.show()
        print("  Saved: recession_probability.png")

    # ── Chart 4: Spread History ──────────────────────────────
    def plot_spread_history(self):
        spreads = {
            "spread_2Y10Y_bp": ("2Y–10Y Spread",  ACCENT_COLORS["blue"]),
            "spread_3M10Y_bp": ("3M–10Y Spread",  ACCENT_COLORS["green"]),
            "spread_2Y30Y_bp": ("2Y–30Y Spread",  ACCENT_COLORS["purple"]),
        }
        available = {k: v for k, v in spreads.items()
                     if k in self.full.columns}
        n = len(available)
        if n == 0:
            print("No spread columns available.")
            return

        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle("U.S. Treasury Yield Spreads — Recession Indicators",
                     fontsize=14, fontweight="bold")

        for ax, (col, (label, color)) in zip(axes, available.items()):
            s = self.full[col].resample("ME").last()
            ax.fill_between(s.index, s.values, 0,
                             where=s.values < 0,
                             color=ACCENT_COLORS["red"], alpha=0.45)
            ax.plot(s.index, s.values, color=color, linewidth=1.5)
            ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
            self._shade_recessions(ax)
            ax.set_ylabel(f"{label} (bps)", fontweight="bold", color=color)
            ax.set_title(f"{label} — Negative → Curve Inversion",
                          fontsize=9, color=ACCENT_COLORS["gray"])

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("spread_history.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print("  Saved: spread_history.png")

    # ── Chart 5: 3D Yield Surface (Plotly) ──────────────────
    def plot_3d_surface(self, n_dates: int = 60):
        cols = [c for c in MAT_LABELS if c in self.yields.columns]
        tau  = [MATURITIES[MAT_LABELS.index(c)] for c in cols]

        idx  = np.linspace(0, len(self.yields) - 1, n_dates, dtype=int)
        df   = self.yields[cols].iloc[idx].dropna(how="all")

        X    = np.array(tau)
        Y    = np.arange(len(df))
        Z    = df.values

        date_labels = [d.strftime("%Y-%m") for d in df.index]

        fig = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            colorscale="Viridis",
            colorbar=dict(title="Yield (%)", tickfont=dict(color="white")),
            hovertemplate=(
                "Maturity: %{x:.1f}Y<br>"
                "Date: %{customdata}<br>"
                "Yield: %{z:.2f}%<extra></extra>"
            ),
            customdata=np.array(date_labels)[np.newaxis, :].repeat(len(X), axis=0).T
        )])

        fig.update_layout(
            title=dict(text="U.S. Treasury Yield Curve Surface<br>"
                            "<sup>Time × Maturity × Yield</sup>",
                        font=dict(size=16)),
            scene=dict(
                xaxis=dict(title="Maturity (Years)", color="white"),
                yaxis=dict(title="Date",
                           tickvals=list(range(0, len(df), max(1, len(df)//8))),
                           ticktext=date_labels[::max(1, len(df)//8)],
                           color="white"),
                zaxis=dict(title="Yield (%)", color="white"),
                bgcolor="#0d1117",
                xaxis_backgroundcolor="#0d1117",
                yaxis_backgroundcolor="#0d1117",
                zaxis_backgroundcolor="#0d1117",
                camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="white"),
            width=900, height=650
        )

        fig.write_html("yield_surface_3d.html")
        fig.show()
        print("  Saved: yield_surface_3d.html")

    # ── Chart 6: Animated Yield Curve (Plotly) ──────────────
    def plot_animated_curve(self, n_frames: int = 48):
        cols = [c for c in MAT_LABELS if c in self.yields.columns]
        tau  = [MATURITIES[MAT_LABELS.index(c)] for c in cols]

        monthly = self.yields[cols].resample("QS").first().dropna(how="all")
        monthly = monthly.iloc[-n_frames:]

        frames = []
        for date, row in monthly.iterrows():
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=tau, y=row.values.tolist(),
                    mode="lines+markers",
                    line=dict(color=ACCENT_COLORS["blue"], width=2.5),
                    marker=dict(size=7, color=ACCENT_COLORS["teal"]),
                    name=date.strftime("%Y-%m")
                )],
                name=date.strftime("%Y-%m"),
                layout=go.Layout(
                    title_text=f"U.S. Treasury Yield Curve — {date.strftime('%b %Y')}"
                )
            ))

        fig = go.Figure(
            data=[go.Scatter(
                x=tau, y=monthly.iloc[0].values.tolist(),
                mode="lines+markers",
                line=dict(color=ACCENT_COLORS["blue"], width=2.5),
                marker=dict(size=7, color=ACCENT_COLORS["teal"])
            )],
            frames=frames
        )

        fig.update_layout(
            title="Animated U.S. Treasury Yield Curve",
            xaxis=dict(title="Maturity (Years)", color="white",
                        range=[0, 31]),
            yaxis=dict(title="Yield (%)", color="white",
                        range=[0, max(10, monthly.max().max() + 0.5)]),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="white"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 300},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0},
                                        "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                steps=[dict(args=[[f.name],
                                  {"frame": {"duration": 300},
                                   "mode": "immediate"}],
                             method="animate",
                             label=f.name)
                        for f in frames],
                currentvalue=dict(prefix="Date: ", visible=True),
                x=0.05, y=0, len=0.9
            )],
            width=880, height=520
        )

        fig.write_html("animated_yield_curve.html")
        fig.show()
        print("  Saved: animated_yield_curve.html")

    # ── Chart 7: Forward Rate Curves ────────────────────────
    def plot_forward_curves(self, forward_dict: Dict):
        if not forward_dict:
            print("No forward curve data available.")
            return

        fig, ax = plt.subplots(figsize=(13, 6))
        fig.suptitle("Instantaneous Forward Rate Curves — Key Historical Dates",
                     fontsize=14, fontweight="bold")

        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(forward_dict)))
        tau_fine = np.linspace(0.08, 30, 300)

        for (date, fwd), color in zip(forward_dict.items(), colors):
            ax.plot(tau_fine, fwd, color=color,
                    linewidth=1.8, label=date.strftime("%b %Y"))

        ax.axhline(0, color=ACCENT_COLORS["gray"],
                   linewidth=0.6, linestyle="--")
        ax.set_xlabel("Maturity (Years)", fontweight="bold")
        ax.set_ylabel("Instantaneous Forward Rate (%)", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_xlim(0, 30)

        plt.tight_layout()
        plt.savefig("forward_curves.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print("  Saved: forward_curves.png")

    # ── Chart 8: ROC Curve ──────────────────────────────────
    def plot_roc_curve(self, probit_model: "RecessionProbitModel"):
        X_sm  = sm.add_constant(probit_model._X_sc, has_constant="add")
        preds = probit_model.result.predict(X_sm)
        y     = probit_model._y.values

        fpr, tpr, _ = roc_curve(y, preds)
        auc         = roc_auc_score(y, preds)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=ACCENT_COLORS["blue"],
                linewidth=2.2, label=f"Probit (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color=ACCENT_COLORS["gray"],
                linewidth=1.0, linestyle="--", label="Random Classifier")
        ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT_COLORS["blue"])
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title("ROC Curve — Probit Recession Model",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig("roc_curve.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print("  Saved: roc_curve.png")

    # ── Run all charts ───────────────────────────────────────
    def run_all(self, probit_model, forward_dict):
        print("\n📈 Generating all visualisations…\n")
        self.plot_ns_factors()
        self.plot_model_fit()
        self.plot_recession_probability()
        self.plot_spread_history()
        self.plot_3d_surface()
        self.plot_animated_curve()
        self.plot_forward_curves(forward_dict)
        self.plot_roc_curve(probit_model)
        print("\n✅ All charts generated.")


# ── Instantiate dashboard and render ────────────────────────
dashboard = YieldCurveDashboard(
    yields    = monthly_yields,
    ns_factors= ns_factors,
    kf_results= kf_results,
    full_df   = full_df,
    rec_proba = rec_proba,
    ns_model  = ns_model
)
dashboard.run_all(probit_model=probit, forward_dict=forward_curves)


# ════════════════════════════════════════════════════════════
# CELL 11: PERFORMANCE METRICS & BACKTESTING
# ════════════════════════════════════════════════════════════

class RecessionSignalBacktest:
    """
    Backtests the yield-curve recession signal against NBER dates.

    Signal definition
    ─────────────────
    • Signal fires when P(recession) ≥ threshold (default 0.30)
    • Lead time measured as months between signal and NBER start

    Metrics
    ───────
    • Hit rate   : fraction of recessions predicted ≥ 1 month ahead
    • False alarm: fraction of signals NOT followed by a recession
    • Lead time  : average months between signal and recession start
    • Brier score: mean-squared error of probability forecasts
    • Regime     : current state (expansion / elevated / recession)
    """

    def __init__(self, rec_proba: pd.Series, full_df: pd.DataFrame,
                 threshold: float = 0.30):
        self.rec_proba = rec_proba
        self.full_df   = full_df
        self.threshold = threshold

    def _get_recession_dates(self):
        if "RECESSION" not in self.full_df.columns:
            return []
        rec     = self.full_df["RECESSION"].resample("ME").last().ffill()
        starts  = []
        in_rec  = False
        for date, val in rec.items():
            if val == 1 and not in_rec:
                starts.append(date)
                in_rec = True
            elif val == 0:
                in_rec = False
        return starts

    def run(self) -> Dict:
        rec_starts = self._get_recession_dates()
        signal     = (self.rec_proba >= self.threshold).astype(int)

        hits        = 0
        false_alarms = 0
        lead_times  = []

        # Signal fires when crossing the threshold
        signal_dates = signal[signal.diff() == 1].index.tolist()

        for sig_date in signal_dates:
            future_rec = [r for r in rec_starts if r > sig_date]
            if future_rec:
                lead = (future_rec[0] - sig_date).days / 30.44
                if lead <= 18:  # Within 18-month window
                    hits      += 1
                    lead_times.append(lead)
                else:
                    false_alarms += 1
            else:
                false_alarms += 1

        n_rec    = len(rec_starts)
        hit_rate = hits / n_rec if n_rec > 0 else 0
        falar    = false_alarms / len(signal_dates) if signal_dates else 0

        results = {
            "recessions_identified": n_rec,
            "signals_fired":         len(signal_dates),
            "recessions_predicted":  hits,
            "false_alarms":          false_alarms,
            "hit_rate_pct":          hit_rate * 100,
            "false_alarm_rate_pct":  falar * 100,
            "avg_lead_months":       np.mean(lead_times) if lead_times else 0,
        }

        # Current regime
        latest_prob = self.rec_proba.iloc[-1]
        regime = ("🔴 ELEVATED RECESSION RISK" if latest_prob >= 0.5
                  else "🟡 MODERATE RISK" if latest_prob >= 0.3
                  else "🟢 EXPANSION")

        print("\n" + "═" * 55)
        print("  YIELD CURVE RECESSION SIGNAL — BACKTEST RESULTS")
        print("═" * 55)
        for k, v in results.items():
            label = k.replace("_", " ").title()
            val   = f"{v:.1f}" if isinstance(v, float) else str(v)
            print(f"  {label:<32} {val}")
        print(f"\n  Current Regime: {regime}")
        print(f"  Latest P(recession): {latest_prob:.1%}")
        print("═" * 55)

        return results


# ── Run backtest ────────────────────────────────────────────
backtest = RecessionSignalBacktest(rec_proba, full_df, threshold=0.30)
bt_results = backtest.run()


# ════════════════════════════════════════════════════════════
# CELL 12: DV01 & DURATION ANALYSIS
# ════════════════════════════════════════════════════════════

print("\n📐 DV01 Analysis for Latest Curve")
latest_date = ns_factors.dropna().index[-1]
latest_betas = ns_factors.loc[latest_date, ["beta1","beta2","beta3"]].values
tau_grid = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
dv01_series = ForwardRateCalculator.compute_dv01(
    latest_betas, tau_grid, face_value=1_000_000
)
print(f"\n  DV01 per $1M face value @ {latest_date.date()}")
print(f"  (basis points shift = 1bp = 0.01%)\n")
print(dv01_series.to_string())


# ════════════════════════════════════════════════════════════
# CELL 13: SUMMARY STATISTICS & EXPORT
# ════════════════════════════════════════════════════════════

def generate_summary_report(yields_df, ns_factors, kf_results,
                             rec_proba, bt_results, metrics):
    """Print a concise project summary to console."""

    print("\n" + "═" * 60)
    print("   PROJECT 6 — YIELD CURVE MODELING SUMMARY REPORT")
    print("═" * 60)

    print(f"\n  📅 Data Period : {yields_df.index[0].date()} → "
          f"{yields_df.index[-1].date()}")
    print(f"  📊 Observations: {len(yields_df):,} trading days")
    print(f"  🏦 Maturities  : {[c for c in MAT_LABELS if c in yields_df.columns]}")

    print("\n  NELSON-SIEGEL MODEL")
    print(f"  Median RMSE      : {ns_factors['rmse'].median():.4f}%")
    print(f"  Mean β1 (Level)  : {ns_factors['beta1'].mean():.2f}%")
    print(f"  Mean β2 (Slope)  : {ns_factors['beta2'].mean():.2f}%")
    print(f"  Mean β3 (Curv.)  : {ns_factors['beta3'].mean():.2f}%")

    print("\n  PROBIT RECESSION MODEL")
    print(f"  ROC-AUC          : {metrics['auc']:.4f}")
    print(f"  McFadden R²      : {metrics['pseudo_r2']:.4f}")
    print(f"  Brier Score      : {metrics['brier']:.4f}")

    print("\n  BACKTEST RESULTS")
    for k, v in bt_results.items():
        print(f"  {k.replace('_',' ').title():<32} "
              f"{'%.1f' % v if isinstance(v, float) else v}")

    print("\n  OUTPUT FILES")
    files = [
        "ns_factors.png",
        "ns_model_fit.png",
        "recession_probability.png",
        "spread_history.png",
        "yield_surface_3d.html",
        "animated_yield_curve.html",
        "forward_curves.png",
        "roc_curve.png",
    ]
    for f in files:
        path = f"{f}"
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  [{exists}] {f}")

    print("\n" + "═" * 60)

generate_summary_report(
    yields_df, ns_factors, kf_results,
    rec_proba, bt_results, metrics
)

# ── Export CSV outputs ──────────────────────────────────────
ns_factors.to_csv("ns_factors.csv")
kf_results.to_csv("dns_kalman_factors.csv")
rec_proba.to_frame("recession_probability").to_csv(
    "recession_probability.csv"
)
print("\n✅ CSV files exported.")
print("   • ns_factors.csv")
print("   • dns_kalman_factors.csv")
print("   • recession_probability.csv")
