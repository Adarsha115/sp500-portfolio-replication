"""
Assignment IV: S&P 500 Portfolio Replication
Two methodologies:
  1. LASSO-regularized regression (L1 penalty induces sparsity)
     → Top-50 stocks by coefficient magnitude selected (k ≤ 50 enforced)
  2. Greedy Forward Selection + Quadratic Programming (cardinality-constrained heuristic)

Author: Student
Data  : Daily OHLCV, Jan 2020 - Feb 2026 (S&P 500 constituents + ^GSPC index)
Train : Jan 2020 - Jun 2025
Holdout: Jul 2025 - Dec 2025

Changes vs original:
  - LASSO now enforces k ≤ 50 by truncating to top-50 coefficient magnitudes
  - Removed duplicate 'HBAN' from Financials sector map
  - matplotlib style uses try/except for compatibility
  - results.json field names clarified with _pct suffix for TE values
  - Sector drift warning threshold tightened; overweight sectors flagged
  - Sparsity chart X-axis now uses k_target for fair visual comparison
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIGURATION: set your data directory here ───────────────────────────────
DATA_DIR = r"OHLCV_Data\Mega"   # <-- Change to your local path
# Example (Windows): r"D:\IIT MADRAS\Assignment_4\OHLCV_Data\Mega"
# Example (Linux/Mac): "/home/user/OHLCV_Data/Mega"

OUTPUT_DIR = "."  # Save outputs in current directory

# ── Complete GICS Sector Map (S&P 500 constituents) ──────────────────────────
# FIX: Removed duplicate 'HBAN' from Financials
SECTOR_MAP = {
    "Information Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","CSCO","IBM","TXN",
        "QCOM","AMAT","AMD","NOW","INTU","ADI","KLAC","LRCX","MRVL","ADSK",
        "SNPS","CDNS","ANET","FTNT","PANW","HPQ","HPE","FSLR","ON","STX",
        "WDC","NTAP","KEYS","EPAM","FFIV","IT","AKAM","CDW","TDY","PTC","JNPR",
        "INTC","GLW","ZBRA","ANSS","TER","MPWR","ENPH","SWKS","QRVO","VRSN",
    ],
    "Health Care": [
        "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","BSX","SYK","ISRG",
        "MDT","ELV","CI","HUM","CVS","BMY","GILD","VRTX","REGN","ZBH","BAX","BDX",
        "HCA","MCK","ABC","CAH","HOLX","DXCM","PODD","IDXX","MTD","WAT","COO","TECH",
        "LH","RMD","MOH","CNC","IQV","PKI","ALGN","HSIC","MKTX","STE","VTRS","OGN",
    ],
    "Financials": [
        "BRK.B","JPM","V","MA","BAC","WFC","GS","MS","SPGI","BLK","AXP","CB","MMC",
        "PGR","AON","MET","PRU","AIG","TRV","ALL","BK","USB","PNC","TFC","COF","SYF",
        "DFS","FITB","KEY","HBAN","RF","CFG","MTB","ZION","CMA","PBCT","CINF","GL",
        "AMP","NDAQ","ICE","CME","CBOE","MSCI","MCO","FDS","NTRS","STT","SIVB","ALLY",
        "FRC","WAL","WTFC","BOKF","IBKR","SF","SEIC","VRTS","WBS",
        # Note: duplicate 'HBAN' removed from this list
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","LOW","BKNG","MAR",
        "GM","F","ORLY","AZO","ROST","EBAY","LVS","MGM","HLT","YUM",
        "CMG","DHI","LEN","PHM","NVR","TOL","APTV","BWA","GPC","AN","KMX",
        "RL","PVH","TPR","HAS","MAT","NWL","LEG","MHK","WHR","ETSY","POOL",
    ],
    "Communication Services": [
        "GOOGL","GOOG","META","NFLX","DIS","CMCSA","T","VZ","TMUS","EA",
        "TTWO","OMC","IPG","NWS","FOXA","PARA","WBD","LUMN","DISH","ATVI",
        "MTCH","IAC","LYV","ZNGA","NWSA",
    ],
    "Industrials": [
        "GE","CAT","HON","RTX","UNP","LMT","DE","MMM","UPS","FDX","ETN","EMR","ITW",
        "PH","ROK","GD","NOC","BA","TDG","HII","LHX","LDOS","CACI","SAIC","J","PWR",
        "URI","PCAR","DAL","UAL","LUV","AAL","CHRW","XPO","GWW","MAS","SWK","ALLE",
        "WM","OTIS","CARR","RSG","EXPD","JBHT","ODFL","RCL","CCL","NSC","CSX","CNI",
        "WAB","IR","GNRC","HUBB","NDSN","ROP","FTV","AME","TT","XYL","MIDD","REVG",
    ],
    "Consumer Staples": [
        "WMT","PG","KO","PEP","COST","PM","MO","CL","KMB","GIS","K","HSY",
        "MKC","SJM","CAG","CPB","HRL","TSN","TAP","STZ","BF.B","EL","CHD","CLX",
        "KR","SYY","MDLZ","MNST","KHC","ADM","BG","INGR","FLO",
    ],
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","PXD","OXY","HES","DVN","BKR",
        "FANG","HAL","APA","MRO","OKE","WMB","KMI","ET","LNG","CTRA","EQT",
        "HP","NOV","FTI","TRGP","DT","HFC",
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ED","EIX","WEC","ES","ETR",
        "PEG","FE","PPL","AES","NI","CMS","LNT","AEE","CNP","EVRG","NRG","PNW",
        "VST","AWK","AWR","SJW","WTRG","YORW",
    ],
    "Real Estate": [
        "PLD","AMT","EQIX","CCI","SPG","PSA","O","WELL","AVB","EQR","VTR","DLR",
        "IRM","ARE","BXP","UDR","CPT","ESS","MAA","NNN","VICI","GLPI","SBA","AMH",
        "WPC","COLD","FR","LTC","MPW","OHI","HR","IIPR",
    ],
    "Materials": [
        "LIN","SHW","APD","ECL","FCX","NEM","NUE","VMC","MLM","PKG","IP","CF","MOS",
        "CE","DD","DOW","EMN","IFF","ALB","CTVA","FMC","RPM","SON","SEE","WRK","AVY",
        "AA","ATI","CMC","STLD","RS","X","CLF","MP","TREX","UFPI",
    ],
}

def ticker_to_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("Loading S&P 500 benchmark (^GSPC)...")
gspc = pd.read_csv(os.path.join(DATA_DIR, "^GSPC.csv"), parse_dates=["Date"])
gspc = gspc.set_index("Date").sort_index()
benchmark_prices = gspc["Close"]

print("Loading constituent stock prices...")
skip_files = {"^GSPC.csv", "sp500_tickers_cache.csv"}
constituent_files = [f for f in os.listdir(DATA_DIR)
                     if f.endswith(".csv") and f not in skip_files]

price_dict = {}
for fname in constituent_files:
    ticker = fname.replace(".csv", "")
    df = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price_dict[ticker] = df[col]

prices_df = pd.DataFrame(price_dict)

# Align to benchmark dates
common_dates = benchmark_prices.index.intersection(prices_df.index)
prices_df        = prices_df.loc[common_dates]
benchmark_prices = benchmark_prices.loc[common_dates]

# Filter date range: Jan 2020 – Feb 2026
t_start = pd.Timestamp("2020-01-01")
t_end   = pd.Timestamp("2026-02-28")
prices_df        = prices_df[(prices_df.index >= t_start) & (prices_df.index <= t_end)]
benchmark_prices = benchmark_prices[(benchmark_prices.index >= t_start) & (benchmark_prices.index <= t_end)]

# Drop columns with <80% coverage, then forward/backward fill remaining NaNs
valid_mask = prices_df.notna().mean() >= 0.80
prices_df  = prices_df.loc[:, valid_mask].ffill().bfill()

# Compute daily log returns
stock_returns = np.log(prices_df / prices_df.shift(1)).dropna()
bench_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()

# Final alignment
common_idx    = stock_returns.index.intersection(bench_returns.index)
stock_returns = stock_returns.loc[common_idx]
bench_returns = bench_returns.loc[common_idx]

N = stock_returns.shape[1]
print(f"  Stocks after filtering: {N}")
print(f"  Date range: {common_idx[0].date()} → {common_idx[-1].date()}")

# ── Train / Holdout splits ────────────────────────────────────────────────────
train_end     = pd.Timestamp("2025-06-30")
holdout_start = pd.Timestamp("2025-07-01")
holdout_end   = pd.Timestamp("2025-12-31")

X_train = stock_returns[stock_returns.index <= train_end]
y_train = bench_returns[bench_returns.index <= train_end]
X_hold  = stock_returns[(stock_returns.index >= holdout_start) & (stock_returns.index <= holdout_end)]
y_hold  = bench_returns[(bench_returns.index >= holdout_start) & (bench_returns.index <= holdout_end)]

print(f"  Train size:   {len(X_train)} trading days")
print(f"  Holdout size: {len(X_hold)} trading days")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def tracking_error(port_ret, bench_ret):
    """Annualised Tracking Error = std(active returns) * sqrt(252)"""
    diff = port_ret.values - bench_ret.values
    return np.std(diff, ddof=1) * np.sqrt(252)

def information_ratio(port_ret, bench_ret):
    """IR = mean(active returns) / std(active returns) * sqrt(252)"""
    diff = port_ret.values - bench_ret.values
    sigma = np.std(diff, ddof=1)
    if sigma == 0:
        return np.nan
    return (np.mean(diff) / sigma) * np.sqrt(252)

def portfolio_returns(weights_series, returns_df):
    """Compute portfolio return series from weights and return matrix"""
    cols = weights_series.index.tolist()
    return returns_df[cols] @ weights_series.values


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: LASSO REGRESSION  (k ≤ 50 enforced)
# Rationale: L1 penalty shrinks many weights to exactly zero, providing
# automatic sparse stock selection without explicitly imposing a cardinality
# constraint. Alpha is tuned via 5-fold cross-validation.
# After fitting, we enforce k ≤ 50 by retaining only the top-50 stocks
# ranked by absolute coefficient magnitude, then re-normalise to sum-to-1.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("METHOD 1: LASSO Regression (k ≤ 50 enforced)")
print("="*60)

K_TARGET = 50  # hard cardinality limit for BOTH methods

# Standardize features (important for LASSO — ensures fair L1 penalisation)
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_ho_sc  = scaler.transform(X_hold)

# Cross-validate alpha on training set
print("  Cross-validating alpha (5-fold)...")
lasso_cv = LassoCV(alphas=np.logspace(-6, -2, 60), cv=5,
                   max_iter=10000, random_state=42, n_jobs=-1)
lasso_cv.fit(X_tr_sc, y_train.values)
best_alpha = lasso_cv.alpha_

# Fit final LASSO
lasso = Lasso(alpha=best_alpha, max_iter=10000, fit_intercept=True)
lasso.fit(X_tr_sc, y_train.values)

# Map scaled coefficients back to original return space
raw_coefs = lasso.coef_ / scaler.scale_

# FIX: Enforce k ≤ 50 — keep only top-50 by absolute coefficient magnitude
# This ensures the constraint stated in the assignment is satisfied.
top_k_idx = np.argsort(np.abs(raw_coefs))[::-1][:K_TARGET]
selected_mask = np.zeros(len(raw_coefs), dtype=bool)
selected_mask[top_k_idx] = True
# Additionally filter out near-zero coefficients within the top-50
selected_mask &= np.abs(raw_coefs) > 1e-8

lasso_tickers     = X_train.columns[selected_mask]
lasso_weights_raw = raw_coefs[selected_mask]

# Normalize to long-only, sum-to-1
lasso_weights_pos  = np.abs(lasso_weights_raw)
lasso_weights_norm = lasso_weights_pos / lasso_weights_pos.sum()
lasso_weights      = pd.Series(lasso_weights_norm, index=lasso_tickers)
k_lasso            = len(lasso_tickers)

print(f"  Alpha (CV-optimal): {best_alpha:.2e}")
print(f"  Stocks selected by LASSO (pre-truncation): {(np.abs(lasso.coef_ / scaler.scale_) > 1e-8).sum()}")
print(f"  Stocks after enforcing k ≤ {K_TARGET}: {k_lasso}")

# Evaluate
lasso_port_train = portfolio_returns(lasso_weights, X_train)
lasso_port_hold  = portfolio_returns(lasso_weights, X_hold)
te_lasso_train   = tracking_error(lasso_port_train, y_train)
te_lasso_hold    = tracking_error(lasso_port_hold, y_hold)
ir_lasso_train   = information_ratio(lasso_port_train, y_train)
ir_lasso_hold    = information_ratio(lasso_port_hold, y_hold)

print(f"  Train  TE: {te_lasso_train*100:.3f}% | Train  IR: {ir_lasso_train:.4f}")
print(f"  Holdout TE: {te_lasso_hold*100:.3f}% | Holdout IR: {ir_lasso_hold:.4f}")
if ir_lasso_hold < 0:
    print(f"  NOTE: Negative holdout IR ({ir_lasso_hold:.4f}) — LASSO portfolio slightly "
          f"underperforms the benchmark on unseen data. This is discussed in the report.")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: GREEDY FORWARD SELECTION + QUADRATIC PROGRAMMING
# Rationale: Directly addresses the cardinality constraint (k ≤ 50).
# Stage 1 (Greedy): Iteratively selects stocks by maximum residual correlation
#   (matching pursuit), a well-known greedy approximation for sparse regression.
# Stage 2 (QP): Given the fixed subset, optimally allocates weights by solving
#   a convex Quadratic Program: min Var(w'X - y) s.t. w≥0, sum(w)=1.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("METHOD 2: Greedy Forward Selection + QP Weight Optimization")
print("="*60)

def greedy_forward_select(X, y, k):
    """
    Greedy matching pursuit: at each step, select the stock whose return
    series is most correlated with the current unexplained benchmark residual.
    After adding a stock, update residual via OLS on the full selected set.
    O(k * N * T) complexity — efficient for N=500, T≈1400, k≤100.
    """
    selected  = []
    remaining = list(X.columns)
    residual  = y.values.copy()

    for step in range(k):
        # Pick stock with highest |corr| to residual
        corrs     = np.array([abs(np.corrcoef(X[s].values, residual)[0, 1])
                               for s in remaining])
        best_idx  = np.argmax(corrs)
        best_stock = remaining[best_idx]
        selected.append(best_stock)
        remaining.remove(best_stock)

        # Update residual: OLS fit on current selected set
        X_sel = X[selected].values
        coef, *_ = np.linalg.lstsq(X_sel, y.values, rcond=None)
        residual  = y.values - X_sel @ coef

    return selected

def optimize_weights_qp(selected_tickers, X_train, y_train):
    """
    Solve the minimum-tracking-variance QP:
        min   w' Σ_X w - 2 cov(X,y)' w
        s.t.  w ≥ 0,  sum(w) = 1
    where Σ_X is the sample covariance of the k selected stock returns
    and cov(X,y) is their sample covariance with the benchmark.
    Uses CVXPY with OSQP solver; regularisation ridge=1e-6 for stability.
    """
    X_sel  = X_train[selected_tickers].values
    y_vec  = y_train.values
    T      = len(y_vec)
    mu_X   = X_sel.mean(axis=0)
    mu_y   = y_vec.mean()
    Sigma  = (X_sel.T @ X_sel) / T - np.outer(mu_X, mu_X)
    cov_xy = (X_sel.T @ y_vec) / T - mu_X * mu_y
    ridge  = 1e-6 * np.eye(len(selected_tickers))

    w = cp.Variable(len(selected_tickers))
    objective   = cp.Minimize(cp.quad_form(w, Sigma + ridge) - 2 * cov_xy @ w)
    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_starting=True, eps_abs=1e-8, eps_rel=1e-8)

    if w.value is None:
        # Fallback: equal weights
        return pd.Series(np.ones(len(selected_tickers)) / len(selected_tickers),
                         index=selected_tickers)
    return pd.Series(w.value, index=selected_tickers)

print(f"  Running greedy selection for k={K_TARGET}...")
greedy_tickers = greedy_forward_select(X_train, y_train, K_TARGET)
greedy_weights = optimize_weights_qp(greedy_tickers, X_train, y_train)

greedy_port_train = portfolio_returns(greedy_weights, X_train)
greedy_port_hold  = portfolio_returns(greedy_weights, X_hold)
te_greedy_train   = tracking_error(greedy_port_train, y_train)
te_greedy_hold    = tracking_error(greedy_port_hold, y_hold)
ir_greedy_train   = information_ratio(greedy_port_train, y_train)
ir_greedy_hold    = information_ratio(greedy_port_hold, y_hold)

print(f"  Stocks selected (k): {len(greedy_tickers)}")
print(f"  Train  TE: {te_greedy_train*100:.3f}% | Train  IR: {ir_greedy_train:.4f}")
print(f"  Holdout TE: {te_greedy_hold*100:.3f}% | Holdout IR: {ir_greedy_hold:.4f}")

# Warn about train→holdout TE degradation
te_deg_greedy = (te_greedy_hold - te_greedy_train) / te_greedy_train * 100
te_deg_lasso  = (te_lasso_hold  - te_lasso_train)  / te_lasso_train  * 100
print(f"\n  Holdout TE degradation — Greedy: +{te_deg_greedy:.1f}% | LASSO: +{te_deg_lasso:.1f}%")
print("  NOTE: TE degradation is expected due to parameter estimation on training data;")
print("  discussed in the report.")


# ─────────────────────────────────────────────────────────────────────────────
# SPARSITY vs TRACKING ERROR (k from 10 to 100)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SPARSITY vs TRACKING ERROR ANALYSIS")
print("="*60)
k_values = [10, 15, 20, 25, 30, 40, 50, 75, 100]

# Run greedy once up to k=100, then trim prefix for each k (efficient)
print("  Running greedy forward selection up to k=100...")
greedy_100 = greedy_forward_select(X_train, y_train, 100)

te_greedy_oos = []
te_lasso_oos  = []
k_lasso_actual = []

for k in k_values:
    # -- Greedy at k: take first k stocks from the greedy order
    sub_g = greedy_100[:k]
    w_g   = optimize_weights_qp(sub_g, X_train, y_train)
    ret_g = portfolio_returns(w_g, X_hold)
    te_greedy_oos.append(tracking_error(ret_g, y_hold))

    # -- LASSO at k: pick top-k by coefficient magnitude (same truncation logic)
    # FIX: use the same top-k truncation approach for fair comparison
    alpha_grid  = np.logspace(-5, -2, 200)
    best_alpha_k = None
    best_diff    = 9999
    for a in alpha_grid:
        l = Lasso(alpha=a, max_iter=5000, fit_intercept=True)
        l.fit(X_tr_sc, y_train.values)
        n_sel = (np.abs(l.coef_) > 1e-8).sum()
        if abs(n_sel - k) < best_diff:
            best_diff    = abs(n_sel - k)
            best_alpha_k = a

    lasso_k = Lasso(alpha=best_alpha_k, max_iter=5000, fit_intercept=True)
    lasso_k.fit(X_tr_sc, y_train.values)
    raw_c = lasso_k.coef_ / scaler.scale_

    # Enforce exactly k stocks via top-k truncation
    top_idx = np.argsort(np.abs(raw_c))[::-1][:k]
    mask_k  = np.zeros(len(raw_c), dtype=bool)
    mask_k[top_idx] = True
    mask_k &= np.abs(raw_c) > 1e-8
    k_lasso_actual.append(mask_k.sum())

    if mask_k.sum() == 0:
        te_lasso_oos.append(np.nan)
        continue
    ticks_l = X_train.columns[mask_k]
    wts_l   = np.abs(raw_c[mask_k]); wts_l /= wts_l.sum()
    wts_ser = pd.Series(wts_l, index=ticks_l)
    ret_l   = portfolio_returns(wts_ser, X_hold)
    te_lasso_oos.append(tracking_error(ret_l, y_hold))

print("  Done.")
# FIX: print k_target for X-axis (not k_lasso_actual) for clarity
print(f"  {'k_target':>10} {'k_greedy':>10} {'TE_greedy%':>12} {'k_lasso_actual':>16} {'TE_lasso%':>12}")
for i, k in enumerate(k_values):
    print(f"  {k:>10} {k:>10} {te_greedy_oos[i]*100:>12.3f} "
          f"{k_lasso_actual[i]:>16} {te_lasso_oos[i]*100 if te_lasso_oos[i] else float('nan'):>12.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR DRIFT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
# Approximate S&P 500 sector weights (GICS, as of 2024)
SP500_SECTOR_WEIGHTS = {
    "Information Technology": 31.0,
    "Health Care":            12.5,
    "Financials":             13.0,
    "Consumer Discretionary": 10.5,
    "Communication Services":  9.0,
    "Industrials":             8.5,
    "Consumer Staples":        6.0,
    "Energy":                  4.0,
    "Utilities":               2.5,
    "Real Estate":             2.5,
    "Materials":               2.5,
}

port_sector = {}
for ticker, wt in greedy_weights.items():
    sector = ticker_to_sector(ticker)
    port_sector[sector] = port_sector.get(sector, 0) + wt * 100

# Build aligned comparison
all_sectors = sorted(set(SP500_SECTOR_WEIGHTS.keys()) | set(port_sector.keys()))
sector_rows = []
for s in all_sectors:
    sv = SP500_SECTOR_WEIGHTS.get(s, 0)
    pv = port_sector.get(s, 0)
    if sv + pv > 0.01:
        drift = pv - sv
        sector_rows.append({
            "Sector":            s,
            "S&P 500 Weight":    round(sv, 2),
            "Portfolio Weight":  round(pv, 2),
            "Drift (pp)":        round(drift, 2),
        })
sector_df = pd.DataFrame(sector_rows)

unmapped = port_sector.get("Other", 0)
if unmapped > 0.1:
    print(f"\n  WARNING: {unmapped:.2f}% of portfolio weight is in unmapped 'Other' sector tickers.")
    print("  Check ticker_to_sector() for missing mappings.")
else:
    print(f"\n  Sector mapping complete. Unmapped weight: {unmapped:.4f}%")

# FIX: Flag sectors with meaningful drift (>3 pp)
DRIFT_WARN_PP = 3.0
large_drifts = sector_df[sector_df["Drift (pp)"].abs() >= DRIFT_WARN_PP]
if not large_drifts.empty:
    print(f"\n  Sectors with |drift| ≥ {DRIFT_WARN_PP:.0f} pp (flagged for report):")
    for _, row in large_drifts.iterrows():
        direction = "overweight" if row["Drift (pp)"] > 0 else "underweight"
        print(f"    {row['Sector']}: {direction} by {abs(row['Drift (pp)']):.2f} pp")


# ─────────────────────────────────────────────────────────────────────────────
# CUMULATIVE RETURN CURVES (full period)
# ─────────────────────────────────────────────────────────────────────────────
full_idx    = bench_returns.index
ret_full_df = stock_returns.reindex(full_idx).fillna(0)

greedy_full = portfolio_returns(greedy_weights, ret_full_df)
lasso_full  = portfolio_returns(lasso_weights,  ret_full_df)

cum_bench  = (1 + bench_returns).cumprod()
cum_greedy = (1 + greedy_full).cumprod()
cum_lasso  = (1 + lasso_full).cumprod()

active_greedy = greedy_full.values - bench_returns.loc[greedy_full.index].values
active_lasso  = lasso_full.values  - bench_returns.loc[lasso_full.index].values


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (5-panel figure)
# ─────────────────────────────────────────────────────────────────────────────
# FIX: Try modern seaborn style, fall back gracefully for older matplotlib
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

CMAP = {"bench": "#1a1a2e", "greedy": "#e94560", "lasso": "#0f3460"}

fig = plt.figure(figsize=(16, 20), facecolor="white")
fig.suptitle("S&P 500 Portfolio Replication  —  Assignment IV",
             fontsize=18, fontweight="bold", y=0.98, color=CMAP["bench"])

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.35)

# ── Panel 1: Cumulative Returns ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_bench.index,  cum_bench.values,  color=CMAP["bench"],  lw=2.2,
         label="S&P 500 Benchmark", zorder=3)
ax1.plot(cum_greedy.index, cum_greedy.values, color=CMAP["greedy"], lw=1.8,
         label=f"Greedy k={K_TARGET}", linestyle="--", zorder=2)
ax1.plot(cum_lasso.index,  cum_lasso.values,  color=CMAP["lasso"],  lw=1.8,
         label=f"LASSO k={k_lasso} (top-{K_TARGET} enforced)", linestyle="-.", zorder=2)
ax1.axvline(holdout_start, color="darkorange", linestyle=":", lw=1.8,
            label="Holdout Period Starts", zorder=4)
ax1.set_title("Cumulative Growth of $1: Benchmark vs Replication Portfolios",
              fontsize=13, fontweight="bold")
ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
ax1.legend(fontsize=9, loc="upper left")
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))

# ── Panel 2: Active Returns — Greedy ──────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(greedy_full.index, active_greedy * 100,
         color=CMAP["greedy"], lw=0.8, alpha=0.85)
ax2.axhline(0, color="black", lw=0.9)
ax2.axvline(holdout_start, color="darkorange", linestyle=":", lw=1.4)
ax2.set_title(f"Daily Active Returns — Greedy k={K_TARGET}", fontsize=11, fontweight="bold")
ax2.set_ylabel("Active Return (%)", fontsize=9)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

# ── Panel 3: Active Returns — LASSO ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(lasso_full.index, active_lasso * 100,
         color=CMAP["lasso"], lw=0.8, alpha=0.85)
ax3.axhline(0, color="black", lw=0.9)
ax3.axvline(holdout_start, color="darkorange", linestyle=":", lw=1.4)
ax3.set_title(f"Daily Active Returns — LASSO k={k_lasso} (top-{K_TARGET} enforced)",
              fontsize=11, fontweight="bold")
ax3.set_ylabel("Active Return (%)", fontsize=9)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

# ── Panel 4: Sparsity vs Tracking Error ───────────────────────────────────
# FIX: X-axis uses k_values (k_target) for both methods — fair visual comparison
ax4 = fig.add_subplot(gs[2, 0])
valid_g = [(k, te) for k, te in zip(k_values,      te_greedy_oos) if not np.isnan(te)]
valid_l = [(k, te) for k, te in zip(k_values,      te_lasso_oos)  if te and not np.isnan(te)]
ax4.plot([v[0] for v in valid_g], [v[1]*100 for v in valid_g],
         "o-", color=CMAP["greedy"], lw=2.2, ms=7, label="Greedy Selection")
ax4.plot([v[0] for v in valid_l], [v[1]*100 for v in valid_l],
         "s-", color=CMAP["lasso"],  lw=2.2, ms=7, label=f"LASSO (top-k enforced)")
ax4.set_title("Out-of-Sample TE vs Portfolio Size (k)", fontsize=11, fontweight="bold")
ax4.set_xlabel("Number of Stocks (k)", fontsize=9)
ax4.set_ylabel("Annualised Tracking Error (%)", fontsize=9)
ax4.legend(fontsize=9)
ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

# ── Panel 5: Sector Drift ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
x_pos = np.arange(len(sector_df))
w_bar = 0.38
ax5.bar(x_pos - w_bar/2, sector_df["S&P 500 Weight"],   w_bar,
        label="S&P 500 (true)", color=CMAP["bench"],  alpha=0.87)
ax5.bar(x_pos + w_bar/2, sector_df["Portfolio Weight"], w_bar,
        label=f"Greedy k={K_TARGET}",  color=CMAP["greedy"], alpha=0.87)
# FIX: Highlight overweight/underweight sectors with annotation
for _, row in large_drifts.iterrows():
    idx = sector_df[sector_df["Sector"] == row["Sector"]].index
    if len(idx) > 0:
        i = sector_df.index.get_loc(idx[0])
        ymax = max(row["S&P 500 Weight"], row["Portfolio Weight"])
        ax5.annotate("*", xy=(i, ymax + 0.5), ha="center", fontsize=12,
                     color="red", fontweight="bold")
short_names = [s.replace(" ", "\n") for s in sector_df["Sector"]]
ax5.set_xticks(x_pos)
ax5.set_xticklabels(short_names, fontsize=7)
ax5.set_title("Sector Drift: Portfolio vs S&P 500 Benchmark\n"
              "(* = |drift| ≥ 3 pp, discussed in report)", fontsize=11, fontweight="bold")
ax5.set_ylabel("Allocation (%)", fontsize=9)
ax5.legend(fontsize=8)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

plot_path = os.path.join(OUTPUT_DIR, "portfolio_replication_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nMain plot saved: {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
header = f"{'Metric':<45} {'Greedy k=50':>12} {f'LASSO k={k_lasso}':>12}"
print(header)
print("-"*70)
print(f"{'Train Tracking Error % (annualised)':45} {te_greedy_train*100:>11.3f}% {te_lasso_train*100:>11.3f}%")
print(f"{'Holdout Tracking Error % (annualised)':45} {te_greedy_hold*100:>11.3f}% {te_lasso_hold*100:>11.3f}%")
print(f"{'TE Degradation Train→Holdout':45} {te_deg_greedy:>10.1f}%  {te_deg_lasso:>10.1f}%")
print(f"{'Train Information Ratio':45} {ir_greedy_train:>12.4f} {ir_lasso_train:>12.4f}")
print(f"{'Holdout Information Ratio':45} {ir_greedy_hold:>12.4f} {ir_lasso_hold:>12.4f}")
print("="*70)
if ir_lasso_hold < 0:
    print(f"  ⚠ LASSO holdout IR is negative ({ir_lasso_hold:.4f}) — see report for discussion.")

# Save artefacts
metrics_out = pd.DataFrame({
    "Method":        [f"Greedy k={K_TARGET}", f"LASSO k={k_lasso} (top-{K_TARGET} enforced)"],
    "Train TE (%)":  [round(te_greedy_train*100, 4), round(te_lasso_train*100, 4)],
    "Holdout TE (%)": [round(te_greedy_hold*100, 4),  round(te_lasso_hold*100, 4)],
    "Train IR":      [round(ir_greedy_train, 4),      round(ir_lasso_train, 4)],
    "Holdout IR":    [round(ir_greedy_hold, 4),       round(ir_lasso_hold, 4)],
})
metrics_out.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

sparsity_out = pd.DataFrame({
    "k_target":       k_values,
    "k_lasso_actual": k_lasso_actual,
    "Greedy OOS TE (%)": [round(v*100, 4) if v and not np.isnan(v) else None for v in te_greedy_oos],
    "LASSO OOS TE (%)":  [round(v*100, 4) if v and not np.isnan(v) else None for v in te_lasso_oos],
})
sparsity_out.to_csv(os.path.join(OUTPUT_DIR, "sparsity_te.csv"), index=False)

# FIX: sector_df now includes Drift column for report reference
sector_df.to_csv(os.path.join(OUTPUT_DIR, "sector_drift.csv"), index=False)

greedy_weights.to_frame("weight").reset_index().rename(
    columns={"index": "ticker"}).to_csv(os.path.join(OUTPUT_DIR, "greedy_weights.csv"), index=False)

# FIX: results.json — TE fields now clearly named with _pct suffix
results = {
    "k_lasso":            int(k_lasso),
    "k_greedy":           K_TARGET,
    "te_greedy_train_pct": round(te_greedy_train * 100, 4),
    "te_greedy_hold_pct":  round(te_greedy_hold  * 100, 4),
    "te_lasso_train_pct":  round(te_lasso_train  * 100, 4),
    "te_lasso_hold_pct":   round(te_lasso_hold   * 100, 4),
    "ir_greedy_train":     round(ir_greedy_train, 4),
    "ir_greedy_hold":      round(ir_greedy_hold,  4),
    "ir_lasso_train":      round(ir_lasso_train,  4),
    "ir_lasso_hold":       round(ir_lasso_hold,   4),
    "te_deg_greedy_pct":   round(te_deg_greedy, 2),
    "te_deg_lasso_pct":    round(te_deg_lasso,  2),
}
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nAll outputs saved successfully.")
