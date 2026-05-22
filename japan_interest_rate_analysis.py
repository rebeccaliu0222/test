
"""
Time series analysis of Japan's central bank policy interest rate.
Data source: FRED (Federal Reserve Bank of St. Louis) — series IRSTCB01JPM156N
"""

import io
import itertools
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ── 1. Fetch data ─────────────────────────────────────────────────────────────
print("Fetching Japan interest rate data from FRED...")
series_id = "IRSTCB01JPM156N"  # Bank of Japan discount rate, % per annum, monthly
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
response = requests.get(url, timeout=30)
response.raise_for_status()
df = pd.read_csv(io.StringIO(response.text), parse_dates=["observation_date"], index_col="observation_date")
df.columns = ["rate"]
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df = df.dropna()
print(f"  Loaded {len(df)} monthly observations: {df.index[0].date()} → {df.index[-1].date()}\n")

# ── 2. Summary statistics ─────────────────────────────────────────────────────
print("── Summary statistics ──────────────────────────────────────")
print(df["rate"].describe().round(3))
print()

# ── 3. ADF stationarity test ──────────────────────────────────────────────────
adf_stat, p_value, _, _, crit_vals, _ = adfuller(df["rate"])
print("── Augmented Dickey-Fuller Test ────────────────────────────")
print(f"  ADF Statistic : {adf_stat:.4f}")
print(f"  p-value       : {p_value:.4f}")
for level, val in crit_vals.items():
    print(f"  Critical ({level})  : {val:.4f}")
conclusion = "stationary (reject unit-root H0)" if p_value < 0.05 else "non-stationary (fail to reject unit-root H0)"
print(f"  -> Series is {conclusion}\n")

# ── 4. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
fig.suptitle("Japan Interest Rate — Time Series Analysis", fontsize=15, fontweight="bold", y=0.98)

date_fmt = mdates.AutoDateLocator()
date_label = mdates.ConciseDateFormatter(date_fmt)

# --- Panel 1: Raw series with rolling statistics ---
ax1 = fig.add_subplot(4, 1, 1)
rolling = df["rate"].rolling(window=24)
ax1.plot(df.index, df["rate"], color="#2c7bb6", linewidth=0.9, label="Monthly rate")
ax1.plot(df.index, rolling.mean(), color="#d7191c", linewidth=1.6, label="24-month rolling mean")
ax1.fill_between(
    df.index,
    rolling.mean() - rolling.std(),
    rolling.mean() + rolling.std(),
    alpha=0.2, color="#d7191c", label="±1 std dev"
)
ax1.set_title("Policy Interest Rate with Rolling Mean ± 1 Std Dev")
ax1.set_ylabel("Rate (%)")
ax1.legend(fontsize=8)
ax1.xaxis.set_major_locator(date_fmt)
ax1.xaxis.set_major_formatter(date_label)
ax1.grid(alpha=0.3)

# --- Panel 2–4: STL/classical decomposition (trend, seasonality, residual) ---
# Resample to ensure regular monthly frequency
df_monthly = df["rate"].resample("MS").mean().interpolate()
decomp = seasonal_decompose(df_monthly, model="additive", period=12)

for i, (component, title, color) in enumerate(
    [
        (decomp.trend, "Trend", "#1a9641"),
        (decomp.seasonal, "Seasonality (period = 12 months)", "#756bb1"),
        (decomp.resid, "Residuals", "#d95f02"),
    ],
    start=2,
):
    ax = fig.add_subplot(4, 1, i)
    ax.plot(component.index, component, color=color, linewidth=0.8)
    if title.startswith("Residuals"):
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Rate (%)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("japan_rate_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 2: ACF / PACF ---
fig2, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(14, 7))
fig2.suptitle("Japan Interest Rate — Autocorrelation Analysis", fontsize=13, fontweight="bold")

plot_acf(df["rate"], ax=ax_acf, lags=48, alpha=0.05)
ax_acf.set_title("Autocorrelation Function (ACF) — up to 48 lags")
ax_acf.set_xlabel("Lag (months)")
ax_acf.grid(alpha=0.3)

plot_pacf(df["rate"], ax=ax_pacf, lags=48, alpha=0.05, method="ywm")
ax_pacf.set_title("Partial Autocorrelation Function (PACF) — up to 48 lags")
ax_pacf.set_xlabel("Lag (months)")
ax_pacf.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("japan_rate_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()

print("Charts saved: japan_rate_overview.png  |  japan_rate_acf_pacf.png")

# ── 5. ARIMA forecast ─────────────────────────────────────────────────────────
print("\nFitting ARIMA model (grid search over p,q in 0-3, d=1)...")
best_aic = np.inf
best_order = (1, 1, 1)
best_result = None

for p, q in itertools.product(range(4), range(4)):
    try:
        res = ARIMA(df_monthly, order=(p, 1, q)).fit()
        if res.aic < best_aic:
            best_aic = res.aic
            best_order = (p, 1, q)
            best_result = res
    except Exception:
        continue

print(f"  Best order : ARIMA{best_order}  (AIC={best_aic:.2f})")

FORECAST_STEPS = 12
fc = best_result.get_forecast(steps=FORECAST_STEPS)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)   # 95% confidence interval

last_date = df_monthly.index[-1]
fc_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=FORECAST_STEPS, freq="MS")
fc_mean.index = fc_index
fc_ci.index = fc_index

print(f"\n  12-month forecast ({fc_index[0].strftime('%b %Y')} - {fc_index[-1].strftime('%b %Y')}):")
print(f"  {'Month':<12}  {'Forecast':>10}  {'Lower 95%':>10}  {'Upper 95%':>10}")
for date, val, lo, hi in zip(fc_index, fc_mean, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1]):
    print(f"  {date.strftime('%b %Y'):<12}  {val:>10.3f}  {lo:>10.3f}  {hi:>10.3f}")

# --- Figure 3: Forecast chart ---
LOOKBACK_YEARS = 5
lookback_start = last_date - pd.DateOffset(years=LOOKBACK_YEARS)
hist = df_monthly[df_monthly.index >= lookback_start]

fig3, ax = plt.subplots(figsize=(14, 6))
fig3.suptitle(
    f"Japan Interest Rate — 12-Month Forecast  (ARIMA{best_order}, AIC={best_aic:.1f})",
    fontsize=13, fontweight="bold"
)

ax.plot(hist.index, hist.values, color="#2c7bb6", linewidth=1.5, label="Historical rate")
ax.plot(fc_mean.index, fc_mean.values, color="#d7191c", linewidth=2, linestyle="--", label="Forecast (mean)")
ax.fill_between(
    fc_ci.index,
    fc_ci.iloc[:, 0],
    fc_ci.iloc[:, 1],
    color="#d7191c", alpha=0.15, label="95% confidence interval"
)
ax.axvline(last_date, color="grey", linewidth=1, linestyle=":", label="Forecast origin")

ax.set_ylabel("Rate (%)")
ax.set_xlabel("")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("japan_rate_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved: japan_rate_forecast.png")
