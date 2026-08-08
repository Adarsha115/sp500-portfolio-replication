"""
Assignment IV: S&P 500 Portfolio Replication
Uses two methodologies:
  1. LASSO-regularized regression (sparse weight selection)
  2. Greedy Forward Selection (heuristic cardinality-constrained)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Sector mapping (GICS) for S&P 500 constituents ───────────────────────────
SECTOR_MAP = {
    "Information Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","CSCO","IBM","TXN",
                                 "QCOM","AMAT","AMD","NOW","INTU","ADI","KLAC","LRCX","MRVL","ADSK",
                                 "SNPS","CDNS","ANET","FTNT","PANW","HPQ","HPE","FSLR","ON","STX",
                                 "WDC","NTAP","KEYS","EPAM","FFIV","IT","AKAM","CDW","TDY","PTC","JNPR"],
    "Health Care": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","BSX","SYK","ISRG",
                    "MDT","ELV","CI","HUM","CVS","BMY","GILD","VRTX","REGN","ZBH","BAX","BDX",
                    "HCA","MCK","ABC","CAH","HOLX","DXCM","PODD","IDXX","MTD","WAT","COO","TECH"],
    "Financials": ["BRK.B","JPM","V","MA","BAC","WFC","GS","MS","SPGI","BLK","AXP","CB","MMC",
                   "PGR","AON","MET","PRU","AIG","TRV","ALL","BK","USB","PNC","TFC","COF","SYF",
                   "DFS","FITB","KEY","HBAN","RF","CFG","MTB","ZION","CMA","PBCT","CINF","GL"],
    "Consumer Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","LOW","BKNG","MAR",
                                 "GM","F","ORLY","AZO","ROST","EBAY","LVS","MGM","HLT","YUM",
                                 "CMG","DHI","LEN","PHM","NVR","TOL","APTV","BWA","GPC","AN","KMX"],
    "Communication Services": ["GOOGL","GOOG","META","NFLX","DIS","CMCSA","T","VZ","TMUS","EA",
                                 "TTWO","OMC","IPG","NWS","FOXA","PARA","WBD","LUMN","DISH"],
    "Industrials": ["GE","CAT","HON","RTX","UNP","LMT","DE","MMM","UPS","FDX","ETN","EMR","ITW",
                    "PH","ROK","GD","NOC","BA","TDG","HII","LHX","LDOS","CACI","SAIC","J","PWR",
                    "URI","PCAR","DAL","UAL","LUV","AAL","CHRW","XPO","GWW","MAS","SWK","ALLE"],
    "Consumer Staples": ["WMT","PG","KO","PEP","COST","PM","MO","CL","KMB","GIS","K","HSY",
                          "MKC","SJM","CAG","CPB","HRL","TSN","TAP","STZ","BF.B","EL","CHD","CLX"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","PXD","OXY","HES","DVN","BKR",
                "FANG","HAL","APA","MRO","OKE","WMB","KMI","ET","LNG","CTRA","EQT"],
    "Utilities": ["NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ED","EIX","WEC","ES","ETR",
                   "PEG","FE","PPL","AES","NI","CMS","LNT","AEE","CNP","EVRG","NRG","PNW"],
    "Real Estate": ["PLD","AMT","EQIX","CCI","SPG","PSA","O","WELL","AVB","EQR","VTR","DLR",
                     "IRM","ARE","BXP","UDR","CPT","ESS","MAA","NNN","VICI","GLPI","SBA","AMH"],
    "Materials": ["LIN","SHW","APD","ECL","FCX","NEM","NUE","VMC","MLM","PKG","IP","CF","MOS",
                   "CE","DD","DOW","EMN","IFF","ALB","CTVA","FMC","RPM","SON","SEE","WRK","AVY"],
}

def ticker_to_sector(ticker):
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = r"D:\IIT MADRAS MTech\2ND SEM\DSAI in Finance\Assignments\Assignment_4\OHLCV_Data\Mega"

print("Loading benchmark (^GSPC)...")
gspc = pd.read_csv(os.path.join(DATA_DIR, "^GSPC.csv"), parse_dates=["Date"])
gspc = gspc.set_index("Date").sort_index()
benchmark_prices = gspc["Close"]

print("Loading constituent prices...")
constituent_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f != "^GSPC.csv" and f != "sp500_tickers_cache.csv"]
price_dict = {}
for fname in constituent_files:
    ticker = fname.replace(".csv", "")
    df = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if "Adj Close" in df.columns:
        price_dict[ticker] = df["Adj Close"]
    elif "Close" in df.columns:
        price_dict[ticker] = df["Close"]

prices_df = pd.DataFrame(price_dict)

# Align to benchmark dates
common_dates = benchmark_prices.index.intersection(prices_df.index)
prices_df = prices_df.loc[common_dates]
benchmark_prices = benchmark_prices.loc[common_dates]

# Filter: Jan 2020 – Feb 2026
t_start = pd.Timestamp("2020-01-01")
t_end   = pd.Timestamp("2026-02-28")
prices_df        = prices_df[(prices_df.index >= t_start) & (prices_df.index <= t_end)]
benchmark_prices = benchmark_prices[(benchmark_prices.index >= t_start) & (benchmark_prices.index <= t_end)]

# Drop cols with too many NaNs (< 80% data), then forward-fill rest
threshold = 0.80
valid = prices_df.notna().mean() >= threshold
prices_df = prices_df.loc[:, valid].ffill().bfill()

# Daily log returns
stock_returns = np.log(prices_df / prices_df.shift(1)).dropna()
bench_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()

# Align
common_idx = stock_returns.index.intersection(bench_returns.index)
stock_returns = stock_returns.loc[common_idx]
bench_returns = bench_returns.loc[common_idx]

print(f"Stocks available after filtering: {stock_returns.shape[1]}")
print(f"Date range: {common_idx[0].date()} → {common_idx[-1].date()}")

# ── Train / Validation / Holdout splits ──────────────────────────────────────
train_end     = pd.Timestamp("2025-06-30")
holdout_start = pd.Timestamp("2025-07-01")
holdout_end   = pd.Timestamp("2025-12-31")

X_train = stock_returns[stock_returns.index <= train_end]
y_train = bench_returns[bench_returns.index <= train_end]

X_hold = stock_returns[(stock_returns.index >= holdout_start) & (stock_returns.index <= holdout_end)]
y_hold = bench_returns[(bench_returns.index >= holdout_start) & (bench_returns.index <= holdout_end)]

print(f"\nTrain size : {len(X_train)} days")
print(f"Holdout size: {len(X_hold)} days")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def tracking_error(port_ret, bench_ret):
    diff = port_ret.values - bench_ret.values
    return np.std(diff, ddof=1) * np.sqrt(252)

def information_ratio(port_ret, bench_ret):
    diff = port_ret.values - bench_ret.values
    if np.std(diff, ddof=1) == 0:
        return np.nan
    return np.mean(diff) / np.std(diff, ddof=1) * np.sqrt(252)

def portfolio_returns(weights_series, returns_df):
    cols = weights_series.index
    return returns_df[cols] @ weights_series.values


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: LASSO-Based Sparse Replication
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Method 1: LASSO Regression ──")

# Scale X for LASSO stability
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_ho_sc = scaler.transform(X_hold)

# Choose alpha via cross-val on training set
lasso_cv = LassoCV(alphas=np.logspace(-6, -2, 50), cv=5, max_iter=5000, random_state=42)
lasso_cv.fit(X_tr_sc, y_train.values)
best_alpha = lasso_cv.alpha_

lasso = Lasso(alpha=best_alpha, max_iter=5000, positive=False, fit_intercept=True)
lasso.fit(X_tr_sc, y_train.values)

raw_coefs = lasso.coef_
# Map back to original scale
lasso_raw_weights = raw_coefs / scaler.scale_
selected_mask = np.abs(lasso_raw_weights) > 1e-6
lasso_tickers = X_train.columns[selected_mask]
lasso_weights_raw = lasso_raw_weights[selected_mask]

# Normalize weights to sum to 1 (long-only reweight)
lasso_weights_pos = np.abs(lasso_weights_raw)
lasso_weights_norm = lasso_weights_pos / lasso_weights_pos.sum()
lasso_weights = pd.Series(lasso_weights_norm, index=lasso_tickers)

k_lasso = len(lasso_tickers)
print(f"  LASSO selected k={k_lasso} stocks (alpha={best_alpha:.6f})")

# Performance
lasso_port_train = portfolio_returns(lasso_weights, X_train)
lasso_port_hold  = portfolio_returns(lasso_weights, X_hold)

te_lasso_train = tracking_error(lasso_port_train, y_train)
te_lasso_hold  = tracking_error(lasso_port_hold, y_hold)
ir_lasso_train = information_ratio(lasso_port_train, y_train)
ir_lasso_hold  = information_ratio(lasso_port_hold, y_hold)

print(f"  Train TE: {te_lasso_train:.4f} | Holdout TE: {te_lasso_hold:.4f}")
print(f"  Train IR: {ir_lasso_train:.4f} | Holdout IR: {ir_lasso_hold:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: Greedy Forward Selection + CVXPY Weight Optimization
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Method 2: Greedy Forward Selection + Quadratic Weight Optimization ──")

def greedy_forward_select(X, y, k):
    """
    Greedily add stocks one at a time; at each step pick the stock whose
    inclusion reduces the residual variance the most.
    """
    selected = []
    remaining = list(X.columns)
    current_residual = y.copy()

    for _ in range(k):
        best_stock = None
        best_corr = -1
        for s in remaining:
            c = abs(np.corrcoef(X[s].values, current_residual.values)[0, 1])
            if c > best_corr:
                best_corr = c
                best_stock = s
        selected.append(best_stock)
        remaining.remove(best_stock)
        # Update residual: regress current residual on newly added stock
        x_sel = X[selected].values
        coef, *_ = np.linalg.lstsq(x_sel, y.values, rcond=None)
        current_residual = pd.Series(y.values - x_sel @ coef, index=y.index)

    return selected

def optimize_weights_qp(selected_tickers, X_train, y_train):
    """
    Solve minimum tracking variance QP with long-only, sum-to-1 constraints.
    min  Var(w'X - y)  ≡  min w'Σ_X w - 2 cov(X,y)'w
    s.t. w >= 0, sum(w)=1
    """
    X_sel = X_train[selected_tickers].values
    y_vec = y_train.values
    T = len(y_vec)
    Sigma = (X_sel.T @ X_sel) / T - (X_sel.mean(0)[:, None] @ X_sel.mean(0)[None, :])
    cov_xy = (X_sel.T @ y_vec) / T - X_sel.mean(0) * y_vec.mean()

    w = cp.Variable(len(selected_tickers))
    obj = cp.quad_form(w, Sigma + 1e-6 * np.eye(len(selected_tickers))) - 2 * cov_xy @ w
    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.OSQP, warm_starting=True)
    return pd.Series(w.value if w.value is not None else np.ones(len(selected_tickers))/len(selected_tickers),
                     index=selected_tickers)

K_GREEDY = 50
print(f"  Running greedy selection for k={K_GREEDY}...")
greedy_tickers = greedy_forward_select(X_train, y_train, K_GREEDY)
greedy_weights = optimize_weights_qp(greedy_tickers, X_train, y_train)

greedy_port_train = portfolio_returns(greedy_weights, X_train)
greedy_port_hold  = portfolio_returns(greedy_weights, X_hold)

te_greedy_train = tracking_error(greedy_port_train, y_train)
te_greedy_hold  = tracking_error(greedy_port_hold, y_hold)
ir_greedy_train = information_ratio(greedy_port_train, y_train)
ir_greedy_hold  = information_ratio(greedy_port_hold, y_hold)

print(f"  Greedy k={K_GREEDY} | Train TE: {te_greedy_train:.4f} | Holdout TE: {te_greedy_hold:.4f}")
print(f"  Train IR: {ir_greedy_train:.4f} | Holdout IR: {ir_greedy_hold:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SPARSITY vs TRACKING ERROR (both methods, varying k)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Sparsity vs TE Analysis ──")
k_values = [10, 15, 20, 25, 30, 40, 50, 75, 100]

te_greedy_oos = []
te_lasso_oos  = []

# Precompute greedy subsets for each k using the same greedy order (efficient)
print("  Greedy forward full run (k=100)...")
greedy_100 = greedy_forward_select(X_train, y_train, 100)

for k in k_values:
    # Greedy: take first k from the ordered list
    sub = greedy_100[:k]
    w = optimize_weights_qp(sub, X_train, y_train)
    ret = portfolio_returns(w, X_hold)
    te_greedy_oos.append(tracking_error(ret, y_hold))

    # LASSO: vary alpha to get different sparsity levels
    target_alpha = best_alpha * (100 / k) ** 1.5
    lasso_k = Lasso(alpha=target_alpha, max_iter=5000, positive=False, fit_intercept=True)
    lasso_k.fit(X_tr_sc, y_train.values)
    raw_c = lasso_k.coef_ / scaler.scale_
    mask = np.abs(raw_c) > 1e-6
    k_actual = mask.sum()
    if k_actual == 0:
        te_lasso_oos.append(np.nan)
        continue
    ticks = X_train.columns[mask]
    wts = np.abs(raw_c[mask]); wts /= wts.sum()
    wts_ser = pd.Series(wts, index=ticks)
    ret_l = portfolio_returns(wts_ser, X_hold)
    te_lasso_oos.append(tracking_error(ret_l, y_hold))

print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR DRIFT ANALYSIS (k=50 Greedy portfolio)
# ─────────────────────────────────────────────────────────────────────────────
# True S&P 500 approximate sector weights (as of ~2024)
sp500_sector_weights = {
    "Information Technology": 0.310,
    "Health Care": 0.125,
    "Financials": 0.130,
    "Consumer Discretionary": 0.105,
    "Communication Services": 0.090,
    "Industrials": 0.085,
    "Consumer Staples": 0.060,
    "Energy": 0.040,
    "Utilities": 0.025,
    "Real Estate": 0.025,
    "Materials": 0.025,
    "Other": 0.000,
}

# Portfolio sector weights
port_sector = {}
for ticker, wt in greedy_weights.items():
    sector = ticker_to_sector(ticker)
    port_sector[sector] = port_sector.get(sector, 0) + wt

# Align sectors
all_sectors = sorted(set(list(sp500_sector_weights.keys()) + list(port_sector.keys())))
sp500_vals  = [sp500_sector_weights.get(s, 0) for s in all_sectors]
port_vals   = [port_sector.get(s, 0) for s in all_sectors]

# Remove "Other" row from both if both zero
rows = [(s, sv, pv) for s, sv, pv in zip(all_sectors, sp500_vals, port_vals) if sv+pv > 0.001]
all_sectors, sp500_vals, port_vals = zip(*rows)


# ─────────────────────────────────────────────────────────────────────────────
# CUMULATIVE RETURN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
# Full period for visual
full_idx = bench_returns.index
greedy_full = portfolio_returns(greedy_weights, stock_returns.reindex(full_idx).fillna(0))
lasso_full  = portfolio_returns(lasso_weights,  stock_returns.reindex(full_idx).fillna(0))

cum_bench  = (1 + bench_returns).cumprod()
cum_greedy = (1 + greedy_full).cumprod()
cum_lasso  = (1 + lasso_full).cumprod()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"bench": "#1a1a2e", "greedy": "#e94560", "lasso": "#0f3460", "accent": "#16213e"}
FIG_SIZE = (16, 20)

fig = plt.figure(figsize=FIG_SIZE, facecolor="white")
fig.suptitle("S&P 500 Portfolio Replication — Assignment IV", fontsize=18, fontweight="bold",
             y=0.98, color=COLORS["accent"])

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── Plot 1: Cumulative Returns ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_bench.index,  cum_bench.values,  color=COLORS["bench"],  lw=2,   label="S&P 500 Benchmark", zorder=3)
ax1.plot(cum_greedy.index, cum_greedy.values, color=COLORS["greedy"], lw=1.8, label=f"Greedy k=50", linestyle="--", zorder=2)
ax1.plot(cum_lasso.index,  cum_lasso.values,  color=COLORS["lasso"],  lw=1.8, label=f"LASSO k≈{k_lasso}", linestyle="-.", zorder=2)
ax1.axvline(holdout_start, color="gray", linestyle=":", lw=1.5, label="Holdout Start")
ax1.fill_betweenx([ax1.get_ylim()[0] if ax1.get_ylim()[0] else 0, 3],
                   pd.Timestamp("2025-07-01"), cum_bench.index[-1],
                   alpha=0.05, color="orange", label="_nolegend_")
ax1.set_title("Cumulative Returns: Benchmark vs Replication Portfolios", fontsize=13, fontweight="bold")
ax1.set_ylabel("Growth of $1", fontsize=10)
ax1.legend(fontsize=9, loc="upper left")
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))

# ── Plot 2: Active Returns (Greedy) ───────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
active_greedy_full = greedy_full.values - bench_returns.loc[greedy_full.index].values
ax2.plot(greedy_full.index, active_greedy_full * 100, color=COLORS["greedy"], lw=0.8, alpha=0.8)
ax2.axhline(0, color="black", lw=0.8)
ax2.axvline(holdout_start, color="gray", linestyle=":", lw=1.2)
ax2.set_title("Active Daily Returns — Greedy k=50", fontsize=11, fontweight="bold")
ax2.set_ylabel("Active Return (%)", fontsize=9)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

# ── Plot 3: Active Returns (LASSO) ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
active_lasso_full = lasso_full.values - bench_returns.loc[lasso_full.index].values
ax3.plot(lasso_full.index, active_lasso_full * 100, color=COLORS["lasso"], lw=0.8, alpha=0.8)
ax3.axhline(0, color="black", lw=0.8)
ax3.axvline(holdout_start, color="gray", linestyle=":", lw=1.2)
ax3.set_title(f"Active Daily Returns — LASSO k≈{k_lasso}", fontsize=11, fontweight="bold")
ax3.set_ylabel("Active Return (%)", fontsize=9)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

# ── Plot 4: Sparsity vs TE ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
valid_greedy = [(k, te) for k, te in zip(k_values, te_greedy_oos) if not np.isnan(te)]
valid_lasso  = [(k, te) for k, te in zip(k_values, te_lasso_oos)  if not np.isnan(te)]
ax4.plot([v[0] for v in valid_greedy], [v[1]*100 for v in valid_greedy],
         "o-", color=COLORS["greedy"], lw=2, ms=6, label="Greedy Selection")
ax4.plot([v[0] for v in valid_lasso],  [v[1]*100 for v in valid_lasso],
         "s-", color=COLORS["lasso"],  lw=2, ms=6, label="LASSO")
ax4.set_title("Sparsity vs Out-of-Sample Tracking Error", fontsize=11, fontweight="bold")
ax4.set_xlabel("Number of Stocks (k)", fontsize=9)
ax4.set_ylabel("Annualised Tracking Error (%)", fontsize=9)
ax4.legend(fontsize=9)
ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

# ── Plot 5: Sector Drift ───────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
x_pos = np.arange(len(all_sectors))
width = 0.38
bars1 = ax5.bar(x_pos - width/2, [v*100 for v in sp500_vals], width, label="S&P 500 (true)", color=COLORS["bench"], alpha=0.85)
bars2 = ax5.bar(x_pos + width/2, [v*100 for v in port_vals],  width, label="Greedy k=50 Portfolio", color=COLORS["greedy"], alpha=0.85)
short_names = [s.replace(" ", "\n") for s in all_sectors]
ax5.set_xticks(x_pos)
ax5.set_xticklabels(short_names, fontsize=7, rotation=0)
ax5.set_title("Sector Drift: Portfolio vs S&P 500 Weights", fontsize=11, fontweight="bold")
ax5.set_ylabel("Allocation (%)", fontsize=9)
ax5.legend(fontsize=8)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

plt.savefig(r"D:\IIT MADRAS MTech\2ND SEM\DSAI in Finance\Assignments\Assignment_4\portfolio_replication_analysis.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("\nPlot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY OF RESULTS")
print("="*65)
print(f"{'Metric':<40} {'Greedy k=50':>12} {'LASSO k≈'+str(k_lasso):>12}")
print("-"*65)
print(f"{'Train Tracking Error (ann.)':40} {te_greedy_train*100:>11.3f}% {te_lasso_train*100:>11.3f}%")
print(f"{'Holdout Tracking Error (ann.)':40} {te_greedy_hold*100:>11.3f}% {te_lasso_hold*100:>11.3f}%")
print(f"{'Train Information Ratio':40} {ir_greedy_train:>12.4f} {ir_lasso_train:>12.4f}")
print(f"{'Holdout Information Ratio':40} {ir_greedy_hold:>12.4f} {ir_lasso_hold:>12.4f}")
print("="*65)

# Save metrics to CSV for report
metrics_df = pd.DataFrame({
    "Method": ["Greedy k=50", f"LASSO k≈{k_lasso}"],
    "Train TE": [round(te_greedy_train*100, 4), round(te_lasso_train*100, 4)],
    "Holdout TE": [round(te_greedy_hold*100, 4), round(te_lasso_hold*100, 4)],
    "Train IR": [round(ir_greedy_train, 4), round(ir_lasso_train, 4)],
    "Holdout IR": [round(ir_greedy_hold, 4), round(ir_lasso_hold, 4)],
})
metrics_df.to_csv(r"D:\IIT MADRAS MTech\2ND SEM\DSAI in Finance\Assignments\Assignment_4\metrics.csv", index=False)

# Save sparsity-TE for report
sparsity_df = pd.DataFrame({
    "k": k_values,
    "Greedy OOS TE": [round(v*100,4) if not np.isnan(v) else None for v in te_greedy_oos],
    "LASSO OOS TE":  [round(v*100,4) if not np.isnan(v) else None for v in te_lasso_oos],
})
sparsity_df.to_csv("sparsity_te.csv", index=False)

# Save sector drift
sector_df = pd.DataFrame({
    "Sector": all_sectors,
    "S&P 500 Weight": [round(v*100,2) for v in sp500_vals],
    "Portfolio Weight": [round(v*100,2) for v in port_vals],
})
sector_df.to_csv("sector_drift.csv", index=False)

# Export greedy weights
greedy_weights.to_frame("weight").reset_index().rename(columns={"index":"ticker"}).to_csv(
    "greedy_weights.csv", index=False)

print("\nAll outputs saved.")

# Store key vars for report generation
import json
results = {
    "k_lasso": int(k_lasso),
    "te_greedy_train": round(te_greedy_train*100, 4),
    "te_greedy_hold":  round(te_greedy_hold*100, 4),
    "te_lasso_train":  round(te_lasso_train*100, 4),
    "te_lasso_hold":   round(te_lasso_hold*100, 4),
    "ir_greedy_train": round(ir_greedy_train, 4),
    "ir_greedy_hold":  round(ir_greedy_hold, 4),
    "ir_lasso_train":  round(ir_lasso_train, 4),
    "ir_lasso_hold":   round(ir_lasso_hold, 4),
}
with open("results.json", "w") as f:
    json.dump(results, f)
print("Results JSON saved.")
