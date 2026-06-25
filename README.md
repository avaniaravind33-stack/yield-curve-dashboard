# Yield Curve Modeling, Nelson-Siegel, and Recession Prediction

Models the U.S. Treasury yield curve and predicts recessions 6-12 months ahead using Dynamic Nelson-Siegel and a probit classifier — backtested 1962–present against NBER recession dates.


## Why This Matters

Fixed income is the largest asset class on Earth (~$130T), larger than global equities. Every central bank, sovereign wealth fund, and rates desk — PIMCO, BlackRock Fixed Income, the Federal Reserve itself — models the term structure of interest rates daily to price bonds, manage duration risk, and read the market's expectations for growth and inflation.

Within that term structure, the **2-10 year spread** is the single most-watched recession indicator in macro. An inverted curve (short rates above long rates) has preceded every U.S. recession since 1955, and tracking that signal — rigorously, not anecdotally — is core to how rates desks and macro hedge funds (Bridgewater, Brevan Howard) position duration ahead of turns in the cycle.

This project builds that signal from scratch: parametrizing the full curve, extracting its latent factors, and turning curve shape into a calibrated recession probability.

## Methodology

- **Data**: 30 years of daily U.S. Treasury yields (1M–30Y maturities) pulled from FRED, plus NBER recession dates, unemployment, and CPI as auxiliary series
- **Nelson-Siegel (NS)**: fits each day's curve with three latent factors — Level (β1), Slope (β2), Curvature (β3) — via nonlinear least squares (`scipy.optimize.minimize`)
- **Nelson-Siegel-Svensson (NSS)**: extends NS with a second curvature term for a better fit at the long end of the curve
- **Dynamic Nelson-Siegel (DNS)**: casts the β factors as a state-space model with VAR(1) dynamics, estimated via Kalman Filter, to smooth and forecast the latent factors through time
- **Feature engineering**: 2Y-10Y spread, 3M-10Y spread, and curve curvature as explanatory variables
- **Probit regression**: classifies P(recession in next 12 months) using the engineered curve features
- **Backtest**: full sample 1962–present, signal dates compared directly against official NBER recession start dates

## Results

| Metric | Value |
|---|---|
| Recessions correctly flagged (hit rate) | — |
| False positive rate | — |
| Average lead time before recession onset | — |
| Sample period | 1962–present |

*(Fill in with your backtest output — e.g., "7 of 8 NBER recessions correctly flagged, with an average lead time of 9.4 months and 1 false positive signal.")*

## Stack

`pandas-datareader` / `fredapi` · `scipy` · `numpy` · `statsmodels` · `filterpy` · `scikit-learn` · `matplotlib` · `plotly`

## Repo Structure

```
.
├── figures/                  # Saved plots and animations referenced above
├── data/                     # Cached FRED pulls
├── notebooks/                # Exploration and model development
├── src/                      # NS/NSS calibration, Kalman filter, probit model
└── README.md
```
