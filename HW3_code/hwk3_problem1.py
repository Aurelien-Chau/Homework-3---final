"""
MATH 60231 - Homework 3, Problem 1: Simulating forecast uncertainty (PJM hourly prices).

Run from the folder containing data/PJM.csv:
    python hwk3_problem1.py
Every table and figure in the write-up for Problem 1 is produced by this script
(tables -> output/p1*.csv, figures -> output/fig_p1*.png). Seeds are fixed.

Conventions
-----------
* Sample: the last 8,760 hourly observations of the Zonal COMED price, which run
  from 2017-12-25 00:00 to 2018-12-24 23:00 (the file ends on 24 December 2018,
  so "the last 8,760 hours" is the 2018 calendar year shifted back by one week).
* Arrays of observations are ordered oldest -> newest, as in the homework:
  s = [s_{t-p+1}, ..., s_t]. Coefficients are phi = [phi_0, phi_1, ..., phi_p],
  where phi_j multiplies s_{t-j+1} (lag j).
* AR models are estimated by OLS on the T - p rows that have p lags available.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
SEED = 60231
P, H, NSIM = 24, 48, 1_000
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
})


def load_prices() -> pd.Series:
    """Full hourly Zonal COMED price series, 2013-01-01 to 2018-12-24 (USD/MWh)."""
    df = pd.read_csv(DATA / "PJM.csv", parse_dates=["Date"], index_col="Date")
    df.columns = df.columns.str.strip()                   # headers carry a leading space
    return df["Zonal COMED price"].rename("price")


# =============================================================================
# (a) one-step forecast
# =============================================================================
def Q1_ARForecast(s, phi) -> float:
    """1-step-ahead AR(p) forecast  s_hat_{t+1} = phi_0 + sum_j phi_j s_{t-j+1}.

    s   : the p most recent observations, oldest first [s_{t-p+1}, ..., s_t]
    phi : [phi_0, phi_1, ..., phi_p]
    s[::-1] puts s_t first, so it lines up with phi_1, ..., phi_p.
    """
    s, phi = np.asarray(s, dtype=float), np.asarray(phi, dtype=float)
    return float(phi[0] + phi[1:] @ s[::-1])


# =============================================================================
# (b) OLS estimation
# =============================================================================
def lag_matrix(s, p: int) -> np.ndarray:
    """(T-p) x p matrix whose row for target s_t is [s_{t-1}, ..., s_{t-p}].
    Built with a sliding-window view (no loop over observations)."""
    win = np.lib.stride_tricks.sliding_window_view(np.asarray(s, dtype=float)[:-1], p)
    return win[:, ::-1]                                    # newest lag first = lag 1


def Q1_AREstimate(s, p: int):
    """OLS estimates of an AR(p) with intercept.

    Returns phi = [phi_0, ..., phi_p] ((1+p) array) and the (T-p) in-sample residuals
    e_t = s_t - s_hat_t, for t = p+1, ..., T.
    """
    s = np.asarray(s, dtype=float)
    X = np.column_stack([np.ones(len(s) - p), lag_matrix(s, p)])
    y = s[p:]
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)
    return phi, y - X @ phi


# =============================================================================
# (c) one bootstrap path
# =============================================================================
def Q1_SimPath(bg, s, phi, e, h: int) -> np.ndarray:
    """One simulated path [s_{t+1}, ..., s_{t+h}].

    The AR(p) recursion is iterated forward with Q1_ARForecast; at each step a
    residual is drawn uniformly, with replacement, from e using the BitGenerator bg.
    The same bg keeps advancing across calls, so repeated calls give new paths.
    """
    rng = np.random.Generator(bg)                         # thin wrapper: shares bg's state
    p = len(phi) - 1
    path = np.concatenate([np.asarray(s, dtype=float)[-p:], np.empty(h)])
    shocks = np.asarray(e, dtype=float)[rng.integers(0, len(e), size=h)]
    for k in range(h):                                    # the recursion is inherently sequential
        path[p + k] = Q1_ARForecast(path[k:p + k], phi) + shocks[k]
    return path[p:]


# =============================================================================
# (d) N paths
# =============================================================================
def Q1_SimBands(bg, s, p: int, h: int, N: int) -> np.ndarray:
    """N x h array of bootstrap paths: phi and residuals are estimated once with
    Q1_AREstimate and re-used for every call to Q1_SimPath."""
    s = np.asarray(s, dtype=float)
    phi, e = Q1_AREstimate(s, p)
    return np.vstack([Q1_SimPath(bg, s[-p:], phi, e, h) for _ in range(N)])


def sim_bands_hour_matched(bg, s: pd.Series, p: int, h: int, N: int) -> np.ndarray:
    """Extension for (c)(ii)/(d): same AR(p), but the shock for target hour k is drawn
    only from residuals that belong to the same hour of the day, so the simulated
    volatility follows the intraday pattern. The N paths are iterated together
    (one matrix-vector product per step instead of N calls)."""
    rng = np.random.Generator(bg)
    phi, e = Q1_AREstimate(s.to_numpy(), p)
    hour_e = s.index[p:].hour.to_numpy()
    fut_hours = (s.index[-1] + pd.to_timedelta(np.arange(1, h + 1), unit="h")).hour
    paths = np.tile(np.r_[s.to_numpy()[-p:], np.zeros(h)], (N, 1))
    for k, hr in enumerate(fut_hours):
        pool = e[hour_e == hr]
        shock = pool[rng.integers(0, len(pool), size=N)]
        paths[:, p + k] = phi[0] + paths[:, k:p + k][:, ::-1] @ phi[1:] + shock
    return paths[:, p:]


def point_path(s, phi, h: int) -> np.ndarray:
    """Iterated point forecasts for horizons 1..h (the recursion with zero shocks)."""
    p = len(phi) - 1
    path = list(np.asarray(s, dtype=float)[-p:])
    for _ in range(h):
        path.append(Q1_ARForecast(path[-p:], phi))
    return np.array(path[p:])


def psi_weights(phi, h: int) -> np.ndarray:
    """MA(infinity) weights psi_0..psi_{h-1} of the AR: the h-step forecast error is
    sum_{j<h} psi_j e_{t+h-j}, so its variance is sigma^2 * cumsum(psi^2)."""
    a = np.asarray(phi[1:], dtype=float)
    psi = np.zeros(h); psi[0] = 1.0
    for j in range(1, h):
        k = min(j, len(a))
        psi[j] = a[:k] @ psi[j - 1::-1][:k]
    return psi


# =============================================================================
# (e) rolling forecasts and the DM test
# =============================================================================
def Q2_DM(d, h):
    """Homework 2's DM function, unchanged: the t-statistic of the mean loss
    differential d, with Newey-West (HAC) standard errors using h-1 lags."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all() or len(d) <= h:
        raise ValueError("d must be finite, one dimensional, and longer than h")
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError("h must be a positive integer")
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": h - 1})
    return float(res.tvalues[0])


def rolling_one_step(s: pd.Series, p: int, window: int, first: int) -> pd.Series:
    """Rolling-window 1-step forecasts of s[t] for t = first, ..., len(s)-1.
    The AR(p) is re-estimated with Q1_AREstimate on the `window` observations
    s[t-window], ..., s[t-1] before each forecast (information up to t-1 only)."""
    v = s.to_numpy(dtype=float)
    fc = [Q1_ARForecast(v[t - p:t], Q1_AREstimate(v[t - window:t], p)[0])
          for t in range(first, len(v))]
    return pd.Series(fc, index=s.index[first:])


def day_ahead_hourly(s: pd.Series, p: int, window: int, first: int) -> pd.Series:
    """Fair-information variant: at each midnight origin t (a multiple of 24 after
    `first`), estimate the AR(p) on the previous `window` hours and iterate it 24
    steps ahead, so every hour of the target day is forecast with information up
    to 23:00 of the previous day - the same information set as the daily AR(6)."""
    v = s.to_numpy(dtype=float)
    out = [point_path(v[t - p:t], Q1_AREstimate(v[t - window:t], p)[0], 24)
           for t in range(first, len(v), 24)]
    return pd.Series(np.concatenate(out), index=s.index[first:])


def dm_table(actual, fc_a, fc_b, label):
    """Squared-loss DM comparison of forecast A (hourly) vs B (daily AR(6)).
    d_t = e_A^2 - e_B^2: negative means the hourly model did better."""
    d = (actual - fc_a) ** 2 - (actual - fc_b) ** 2
    tau = len(d)
    dm = Q2_DM(d.to_numpy(), 1)
    mdm = dm * np.sqrt((tau - 1) / tau)                    # HLN factor at h = 1
    dm7 = Q2_DM(d.to_numpy(), 8)                           # robustness: 7 Newey-West lags
    return pd.Series({
        "target days": tau, "first day": d.index[0].date(), "last day": d.index[-1].date(),
        "MSFE hourly AR(24)": ((actual - fc_a) ** 2).mean(),
        "MSFE daily AR(6)": ((actual - fc_b) ** 2).mean(),
        "mean d": d.mean(), "DM": dm, "p (two-sided, N(0,1))": 2 * stats.norm.sf(abs(dm)),
        "MDM (HLN)": mdm, "p (t, tau-1)": 2 * stats.t.sf(abs(mdm), tau - 1),
        "DM, 7 NW lags": dm7, "p, 7 NW lags": 2 * stats.norm.sf(abs(dm7)),
        "share of days hourly better": (d < 0).mean()}, name=label), d


# =============================================================================
# Main
# =============================================================================
def main():
    full = load_prices()
    s = full.iloc[-8760:]
    v = s.to_numpy()
    print("sample", s.index[0], "->", s.index[-1], len(s))

    # ---------------- (b) first, since (a) needs the AR(24) estimates ----------------
    phi, e = Q1_AREstimate(v, P)
    X = sm.add_constant(lag_matrix(v, P))
    ols = sm.OLS(v[P:], X).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=24)
    assert np.allclose(ols.params, phi)                   # our OLS = statsmodels OLS
    coef = pd.DataFrame({"phi": phi, "OLS se": ols.bse, "HAC se": hac.bse,
                         "HAC t": phi / hac.bse},
                        index=["phi_0"] + [f"lag {j}" for j in range(1, P + 1)])
    coef.round(4).to_csv(OUT / "p1b_coefficients.csv")
    fit = pd.Series({"R2": ols.rsquared, "residual sd": e.std(ddof=P + 1),
                     "sum of AR coefs": phi[1:].sum(),
                     "largest modulus root of the AR polynomial": np.abs(np.roots(
                         np.r_[1, -phi[1:]])).max(),
                     "residual skewness": stats.skew(e), "residual excess kurtosis": stats.kurtosis(e)})
    fit.round(4).to_csv(OUT / "p1b_fit.csv")
    print(coef.round(3), "\n", fit.round(3))

    # residual ACF up to two weeks, plus ACF of squared residuals and Ljung-Box tests
    nlag = 336
    acf_e = sm.tsa.acf(e, nlags=nlag, fft=True)
    acf_e2 = sm.tsa.acf(e ** 2, nlags=nlag, fft=True)
    band = 1.96 / np.sqrt(len(e))
    top = pd.Series(acf_e[1:], index=range(1, nlag + 1)).abs().sort_values(ascending=False).head(10)
    lb = sm.stats.acorr_ljungbox(e, lags=[24, 48, 168], return_df=True)
    lb2 = sm.stats.acorr_ljungbox(e ** 2, lags=[24], return_df=True)
    pd.DataFrame({"|acf|": top, "acf": acf_e[top.index]}).round(4).to_csv(OUT / "p1b_resid_acf_top.csv")
    pd.concat([lb.assign(series="resid"), lb2.assign(series="resid^2")]).to_csv(OUT / "p1b_ljungbox.csv")
    print("largest residual autocorrelations\n", top.round(3), "\n", lb, "\n", lb2,
          "\n share of lags 1..336 outside band:", (np.abs(acf_e[1:]) > band).mean().round(3))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].bar(range(1, P + 1), phi[1:], color=COLORS[0], width=0.7)
    axes[0].axhline(0, color="#52514e", lw=0.8)
    axes[0].set(xlabel="lag (hours)", ylabel=r"$\hat\phi_j$", title="AR(24) coefficients",
                xticks=[1, 6, 12, 18, 24])
    axes[1].vlines(range(1, nlag + 1), 0, acf_e[1:], color=COLORS[0], lw=0.8)
    axes[1].axhspan(-band, band, color="#e6e5e0")
    for L in (24, 48, 168, 336):
        axes[1].text(L, acf_e[L] + (0.012 if acf_e[L] >= 0 else -0.02), str(L), ha="center", fontsize=7)
    axes[1].set(xlabel="lag (hours)", title="ACF of AR(24) residuals (band: ±1.96/√T)",
                xticks=[0, 24, 48, 72, 96, 120, 144, 168, 216, 264, 312, 336])
    axes[1].tick_params(axis="x", labelsize=7)
    fig.savefig(OUT / "fig_p1b_coef_acf.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.vlines(range(1, 73), 0, acf_e2[1:73], color=COLORS[1], lw=1)
    ax.axhspan(-band, band, color="#e6e5e0")
    ax.set(xlabel="lag (hours)", title="ACF of squared AR(24) residuals: volatility clustering",
           xticks=range(0, 73, 12))
    fig.savefig(OUT / "fig_p1b_acf_sq.png"); plt.close(fig)

    # ---------------- (a) the one-step forecast ----------------
    fc = Q1_ARForecast(v[-P:], phi)
    alt = {}
    for q in (3, 6, 24):                                    # same exercise with shorter AR
        ph_q, e_q = Q1_AREstimate(v, q)
        n_q = len(e_q)
        alt[f"AR({q})"] = {"forecast for 2018-12-25 00:00": Q1_ARForecast(v[-q:], ph_q),
                           "in-sample RMSE": np.sqrt((e_q ** 2).mean()),
                           # BIC on a common sample would differ slightly; reported per model
                           "BIC": n_q * np.log((e_q ** 2).mean()) + (q + 1) * np.log(n_q)}
    alt = pd.DataFrame(alt).T
    context = pd.Series({"AR(24) forecast": fc, "last observed (2018-12-24 23:00)": v[-1],
                         "same hour previous day (2018-12-24 00:00)": v[-24],
                         "mean price at midnight in sample": s[s.index.hour == 0].mean(),
                         "sample mean": v.mean()})
    alt.round(3).to_csv(OUT / "p1a_models.csv"); context.round(3).to_csv(OUT / "p1a_forecast.csv")
    print("(a)", context.round(2).to_dict(), "\n", alt.round(3))

    # ---------------- (c) three simulated paths ----------------
    bg = np.random.PCG64(SEED)
    paths3 = np.vstack([Q1_SimPath(bg, v[-P:], phi, e, H) for _ in range(3)])
    fut = pd.date_range(s.index[-1] + pd.Timedelta(hours=1), periods=H, freq="h")
    pf = point_path(v, phi, H)
    pd.DataFrame(paths3.T, index=fut, columns=["path 1", "path 2", "path 3"]).assign(point=pf) \
        .round(3).to_csv(OUT / "p1c_paths.csv")
    # daily-cycle check: correlation of each path / the point forecast with the in-sample hourly profile
    prof = s.groupby(s.index.hour).mean()
    cyc = pd.Series({k: np.corrcoef(x, prof.reindex(fut.hour).to_numpy())[0, 1]
                     for k, x in zip(["path 1", "path 2", "path 3", "point forecast"], [*paths3, pf])})
    cyc.round(3).to_csv(OUT / "p1c_cycle_corr.csv")
    print("(c) corr with average hourly profile\n", cyc.round(2))

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    hist = s.iloc[-72:]
    ax.plot(hist.index, hist, color="#0b0b0b", lw=1.3, label="observed (last 72 h)")
    for k in range(3):
        ax.plot(fut, paths3[k], color=COLORS[k], lw=1.1, label=f"simulated path {k + 1}")
    ax.plot(fut, pf, color="#52514e", lw=1, ls="--", label="point forecast")
    ax.axvline(s.index[-1] + pd.Timedelta(minutes=30), color="#52514e", lw=0.9)
    ax.text(s.index[-1], ax.get_ylim()[1], " forecast →", va="top", fontsize=7.5, color="#52514e")
    ax.set(ylabel="USD/MWh", title="COMED price: last 72 hours and three bootstrap paths (AR(24))")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.legend(ncol=3, fontsize=7, loc="lower left")
    fig.savefig(OUT / "fig_p1c_paths.png"); plt.close(fig)

    # ---------------- (d) N = 1,000 paths and the 90% band ----------------
    sims = Q1_SimBands(np.random.PCG64(SEED + 1), v, P, H, NSIM)
    q05, q50, q95 = np.percentile(sims, [5, 50, 95], axis=0)
    width = q95 - q05
    psi = psi_weights(phi, H)
    sigma = e.std()
    gauss_width = 2 * 1.645 * sigma * np.sqrt(np.cumsum(psi ** 2))   # Gaussian band, same phi
    bands = pd.DataFrame({"point": pf, "mean of paths": sims.mean(0), "q05": q05, "median": q50,
                          "q95": q95, "width": width, "Gaussian width": gauss_width,
                          "psi": psi}, index=fut)
    bands.round(3).to_csv(OUT / "p1d_bands.csv")
    sims_hr = sim_bands_hour_matched(np.random.PCG64(SEED + 3), s, P, H, NSIM)
    w_hr = np.diff(np.percentile(sims_hr, [5, 95], axis=0), axis=0)[0]
    bands["width, hour-matched bootstrap"] = w_hr
    bands.round(3).to_csv(OUT / "p1d_bands.csv")
    # same exercise on the full 2013-2018 history
    sims_full = Q1_SimBands(np.random.PCG64(SEED + 2), full.to_numpy(), P, H, NSIM)
    phi_full, e_full = Q1_AREstimate(full.to_numpy(), P)
    w_full = np.diff(np.percentile(sims_full, [5, 95], axis=0), axis=0)[0]
    cmp = pd.DataFrame({
        "2018 sample": [width[0], width[23], width[47], width.mean(), e.std(), stats.kurtosis(e),
                        np.percentile(e, 99) - np.percentile(e, 1), phi[1:].sum()],
        "2013-2018 sample": [w_full[0], w_full[23], w_full[47], w_full.mean(), e_full.std(),
                             stats.kurtosis(e_full), np.percentile(e_full, 99) - np.percentile(e_full, 1),
                             phi_full[1:].sum()]},
        index=["band width h=1", "band width h=24", "band width h=48", "mean band width",
               "residual sd", "residual excess kurtosis", "residual 1%-99% range", "sum of AR coefs"])
    cmp.round(3).to_csv(OUT / "p1d_full_vs_2018.csv")
    # empirical coverage check of the 2018 method: 1-step band vs the in-sample residuals by hour
    by_hour_sd = pd.Series(e, index=s.index[P:]).groupby(s.index[P:].hour).std()
    by_hour_sd.round(3).to_csv(OUT / "p1d_resid_sd_by_hour.csv")
    print("(d) widths h=1,12,24,25,36,48:", width[[0, 11, 23, 24, 35, 47]].round(2),
          "\n hour-matched widths h=1,6,12,18,24,48:", w_hr[[0, 5, 11, 17, 23, 47]].round(2),
          "\n jump ratios", (width[1:] / width[:-1]).round(3)[[22, 23, 24, 25]],
          "\n", cmp.round(2), "\n resid sd by hour\n", by_hour_sd.round(2).to_dict())

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(hist.index, hist, color="#0b0b0b", lw=1.2, label="observed")
    ax.fill_between(fut, q05, q95, color=COLORS[0], alpha=0.18, lw=0, label="90% band (5th-95th pct.)")
    ax.plot(fut, q05, color=COLORS[0], lw=0.9); ax.plot(fut, q95, color=COLORS[0], lw=0.9)
    ax.plot(fut, pf, color=COLORS[1], lw=1.4, label="point forecast")
    ax.plot(fut, q50, color=COLORS[0], lw=1, ls=":", label="median path")
    ax.axvline(s.index[-1] + pd.Timedelta(minutes=30), color="#52514e", lw=0.9)
    ax.set(ylabel="USD/MWh", title=f"AR(24) residual-bootstrap forecast, N = {NSIM:,} paths, h = 48")
    ax.legend(ncol=2, fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.plot(fut, width, color=COLORS[0], marker="o", ms=2.5, label="bootstrap 90% width")
    ax.plot(fut, gauss_width, color="#52514e", lw=1, ls="--", label=r"Gaussian: $2\cdot1.645\,\hat\sigma\sqrt{\sum\psi_j^2}$")
    ax.plot(fut, w_hr, color=COLORS[2], lw=1.1, label="hour-matched bootstrap width")
    ax.plot(fut, w_full, color=COLORS[1], lw=1.1, label="bootstrap width, 2013-2018 sample")
    ax.set(ylabel="band width (USD/MWh)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.savefig(OUT / "fig_p1d_bands.png"); plt.close(fig)

    # ---------------- (e) hourly AR(24) vs daily AR(6), Diebold-Mariano ----------------
    W_DAYS = 300                                            # Homework 2's rolling window
    daily_all = full.resample("D").mean()                   # daily mean price, full history
    daily = s.resample("D").mean()                          # 365 days in the sample

    def run_dm(hourly_series, daily_series, first_day_idx, label):
        """first_day_idx: position (in days) of the first target day in daily_series."""
        first_hour = hourly_series.index.get_loc(daily_series.index[first_day_idx])
        hw = W_DAYS * 24
        fc_1 = rolling_one_step(hourly_series, P, hw, first_hour).resample("D").mean()
        fc_da = day_ahead_hourly(hourly_series, P, hw, first_hour).resample("D").mean()
        fc_d = rolling_one_step(daily_series, 6, W_DAYS, first_day_idx)
        act = daily_series.iloc[first_day_idx:]
        t1, d1 = dm_table(act, fc_1, fc_d, f"{label}: average of 24 one-hour-ahead")
        t2, d2 = dm_table(act, fc_da, fc_d, f"{label}: day-ahead (1-24 h from midnight)")
        return [t1, t2], pd.DataFrame({"actual": act, "hourly 1-step avg": fc_1,
                                       "hourly day-ahead": fc_da, "daily AR(6)": fc_d,
                                       "d (1-step)": d1, "d (day-ahead)": d2})

    # main: estimation and evaluation inside the 8,760-hour sample (first target = day 301)
    tabs_in, fc_in = run_dm(s, daily, W_DAYS, "in-sample windows")
    # robustness: windows may reach back before the sample, so every day of the sample is a target
    tabs_all, fc_all = run_dm(full, daily_all, daily_all.index.get_loc(daily.index[0]), "full-year targets")
    dm_res = pd.DataFrame(tabs_in + tabs_all).T
    dm_res.to_csv(OUT / "p1e_dm.csv"); fc_in.round(3).to_csv(OUT / "p1e_forecasts.csv")
    fc_all.round(3).to_csv(OUT / "p1e_forecasts_fullyear.csv")
    print("(e)\n", dm_res)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    axes[0].plot(fc_in.index, fc_in["actual"], color="#0b0b0b", lw=1.3, label="realized daily mean")
    axes[0].plot(fc_in.index, fc_in["hourly 1-step avg"], color=COLORS[0], label="hourly AR(24), avg of 1-step")
    axes[0].plot(fc_in.index, fc_in["hourly day-ahead"], color=COLORS[2], lw=1, label="hourly AR(24), day-ahead")
    axes[0].plot(fc_in.index, fc_in["daily AR(6)"], color=COLORS[1], label="daily AR(6) (HW2)")
    axes[0].set(ylabel="USD/MWh", title="Daily forecasts, 21 Oct - 24 Dec 2018 (65 target days)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].bar(fc_in.index, fc_in["d (1-step)"], color=COLORS[0], width=0.8)
    axes[1].axhline(0, color="#52514e", lw=0.8)
    axes[1].set(ylabel=r"$d_t=e^2_{AR24}-e^2_{AR6}$", title="loss differential (negative: hourly model better)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.savefig(OUT / "fig_p1e_dm.png"); plt.close(fig)


if __name__ == "__main__":
    main()
