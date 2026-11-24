"""
MATH 60231 - Homework 3, Problem 2: Unit roots on UK gilt yields and the S&P 500.

Run from the folder containing UK_bonds.parquet and "S&P 500 Historical Data.csv":
    python hwk3_problem2.py
Every table and figure in the write-up for Problem 2 is produced by this script
(tables -> output/*.csv, figures -> output/*.png). Seeds are fixed, so the
numbers are reproducible.

Conventions (stated once here, used everywhere below)
-----------------------------------------------------
* ADF regressions are estimated by OLS on the "effective" sample: with a series of
  length n and p lagged differences, the regression has T = n - 1 - p rows.
  This is the T used in the normalized-bias statistic delta = T * beta_hat
  (it is also the T statsmodels' adfuller uses, which we use as a cross-check).
* The ADF tests are left-tailed: we reject H0 (unit root) when the statistic is
  BELOW the critical value. The "90%, 95%, 99%" critical values are therefore the
  10th, 5th and 1st percentiles of the simulated null distribution.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

STAT_NAMES = ["t (const)", "t (trend)", "delta (const)", "delta (trend)"]
SEED = 60231                      # course number: one fixed seed for the whole problem

# ---- plotting style: one palette, recessive axes, used by every figure --------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
})


# =============================================================================
# Data
# =============================================================================
def load_gilts() -> pd.DataFrame:
    """Daily UK gilt yields (in %), 15 maturities, sorted by date."""
    return pd.read_parquet(DATA / "UK_bonds.parquet").sort_index()


def load_log_spx() -> pd.Series:
    """Log S&P 500 close. The csv is newest-first with '4,685.25'-style strings."""
    df = pd.read_csv(DATA / "S&P 500 Historical Data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    price = pd.to_numeric(df["Price"].str.replace(",", ""), errors="raise")
    return np.log(pd.Series(price.to_numpy(), index=df["Date"], name="log S&P 500")).sort_index()


# =============================================================================
# (b) ADF statistics
# =============================================================================
def _adf_batch(Y: np.ndarray, p: int, adjusted: bool = False) -> np.ndarray:
    """Vectorized core of Q2_ADF: the four ADF statistics for EVERY row of Y (M x n).

    For each row y, OLS of  dy_t = a [+ g*t] + beta*y_{t-1} + sum_i phi_i dy_{t-i} + v_t
    is solved through the normal equations, stacked over rows with einsum/solve, so
    thousands of series (bootstrap walks, rolling windows) cost one linear-algebra call
    instead of a Python loop. Returns an M x 4 array in the order of STAT_NAMES.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    M, n = Y.shape
    dy = np.diff(Y, axis=1)
    T = n - 1 - p                                              # rows of each regression
    lhs = dy[:, p:]                                            # dy_t
    lags = np.stack([dy[:, p - i: n - 1 - i] for i in range(1, p + 1)], axis=2)  # dy_{t-i}
    ylag = Y[:, p:-1, None]                                    # y_{t-1}, aligned with dy_t
    one = np.ones((M, T, 1))
    trend = np.broadcast_to(np.arange(1.0, T + 1)[None, :, None], (M, T, 1))
    out = np.empty((M, 4))
    for col, det in enumerate([[one], [one, trend]]):          # constant, then constant + trend
        X = np.concatenate(det + [ylag, lags], axis=2)         # M x T x k
        XtX = np.einsum("mtk,mtl->mkl", X, X)
        coef = np.linalg.solve(XtX, np.einsum("mtk,mt->mk", X, lhs)[..., None])[..., 0]
        resid = lhs - np.einsum("mtk,mk->mt", X, coef)
        s2 = (resid ** 2).sum(axis=1) / (T - X.shape[2])       # OLS error variance
        j = len(det)                                           # position of y_{t-1}
        beta = coef[:, j]
        out[:, col] = beta / np.sqrt(s2 * np.linalg.inv(XtX)[:, j, j])           # t-statistic
        scale = (1 - coef[:, j + 1:].sum(axis=1)) if adjusted else 1.0
        out[:, col + 2] = T * beta / scale                                      # delta = T*beta
    return out


def Q2_ADF(s, p: int, adjusted: bool = False) -> pd.Series:
    """Four ADF statistics with p lagged differences, in the order asked:
    (i) t, constant; (ii) t, constant + trend; (iii) delta, constant; (iv) delta, trend.

    delta = T * beta_hat as defined in the homework (T = rows of the regression).
    With adjusted=True, delta is instead T * beta_hat / (1 - sum(phi_hat)), the
    lag-corrected normalized bias (Hamilton 1994, ch. 17), whose null distribution
    does not depend on the short-run dynamics. Default (False) = homework definition.
    """
    return pd.Series(_adf_batch(np.asarray(s, dtype=float), p, adjusted)[0], index=STAT_NAMES)


# =============================================================================
# (c) Bootstrap critical values
# =============================================================================
def Q2_CVSim(s, N: int, T: int, p: int, rng: np.random.Generator | None = None,
             demean: bool = False) -> pd.DataFrame:
    """Critical values of the four ADF statistics under H0: unit root.

    (i)   N random walks of length T, increments drawn i.i.d. with replacement from
          the observed first differences of s (so the walks inherit the increments'
          mean, i.e. the drift, and their fat tails);
    (ii)  Q2_ADF(., p) on each walk;
    (iii) 10th / 5th / 1st percentiles (left-tailed tests) -> 90 / 95 / 99 % rows.
    demean=True removes the sample mean of the increments first (drift-free null);
    it is used only for the drift diagnostic in part (d).
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    ds = np.diff(np.asarray(s, dtype=float))
    if demean:
        ds = ds - ds.mean()
    walks = np.cumsum(rng.choice(ds, size=(N, T), replace=True), axis=1)  # N x T at once
    stats = _adf_batch(walks, p)        # = Q2_ADF(w, p) for each walk w, in one vectorized call
    cv = np.percentile(stats, [10, 5, 1], axis=0)
    return pd.DataFrame(cv, index=["90%", "95%", "99%"], columns=STAT_NAMES)


def simulate_null(s, N, T, p, rng, demean=False, gaussian=False):
    """Same simulation as Q2_CVSim but returns the full N x 4 draws (for plots and
    for the high-precision reference distributions). gaussian=True draws N(0,1)
    increments: the textbook Dickey-Fuller setting."""
    ds = np.diff(np.asarray(s, dtype=float))
    if demean:
        ds = ds - ds.mean()
    inc = rng.standard_normal((N, T)) if gaussian else rng.choice(ds, size=(N, T))
    return _adf_batch(np.cumsum(inc, axis=1), p)


# =============================================================================
# (e) Rolling ADF
# =============================================================================
def Q2_RollingADF(s: pd.Series, p: int, N: int, return_path: bool = False):
    """Q2_ADF(., p) on every window of N consecutive observations of s.

    Returns a 3 x 4 DataFrame: row 1 the most negative value of each statistic,
    rows 2-3 the first and last date of the window that produced it.
    With return_path=True, also returns the full (n-N+1) x 4 path of statistics,
    indexed by the window's END date (used for the figures).
    """
    windows = np.lib.stride_tricks.sliding_window_view(s.to_numpy(dtype=float), N)  # (n-N+1) x N
    path = pd.DataFrame(_adf_batch(windows, p), index=s.index[N - 1:], columns=STAT_NAMES)
    k = path.to_numpy().argmin(axis=0)                 # window index of each minimum
    res = pd.DataFrame([path.to_numpy()[k, range(4)],
                        s.index[k].date,                # window start
                        s.index[k + N - 1].date],       # window end
                       index=["most negative", "window start", "window end"],
                       columns=STAT_NAMES)
    return (res, path) if return_path else res


def rolling_min_null(s, p, N, reps, rng):
    """Null distribution of the MINIMUM rolling statistic: the min over hundreds of
    overlapping windows is not Dickey-Fuller distributed (multiple testing). We
    bootstrap random walks with the length of s from its increments and record the
    minimum of each statistic over all windows. Returns a reps x 4 array."""
    ds = np.diff(np.asarray(s, dtype=float))
    walks = np.cumsum(rng.choice(ds, size=(reps, len(s))), axis=1)
    return np.vstack([_adf_batch(np.lib.stride_tricks.sliding_window_view(w, N), p).min(axis=0)
                      for w in walks])


# =============================================================================
# Main: every number and figure in the write-up
# =============================================================================
def main():
    rng = np.random.default_rng(SEED)
    gilts, lspx = load_gilts(), load_log_spx()
    y2, y10 = gilts["2Yr"], gilts["10Yr"]

    # ---------------- (a) descriptive statistics and graphs ----------------
    d = gilts.diff().dropna()
    desc = pd.DataFrame({
        "mean": gilts.mean(), "sd": gilts.std(), "min": gilts.min(),
        "P25": gilts.quantile(0.25), "median": gilts.median(), "P75": gilts.quantile(0.75),
        "max": gilts.max(),
        "first": gilts.iloc[0], "last": gilts.iloc[-1],
        "days < 0": (gilts < 0).sum(),
        "sd of daily change (bp)": 100 * d.std(),
        "AC(1) level": gilts.apply(lambda c: c.autocorr(1)),
        "AC(1) change": d.apply(lambda c: c.autocorr(1)),
        "excess kurtosis of change": d.kurt(),
    })
    desc.round(3).to_csv(OUT / "p2a_descriptives.csv")
    neg = pd.DataFrame({c: [gilts.index[gilts[c] < 0].min(), gilts.index[gilts[c] < 0].max(),
                            gilts[c].min(), gilts[c].idxmin()]
                        for c in gilts.columns if (gilts[c] < 0).any()},
                       index=["first negative", "last negative", "minimum", "date of min"]).T
    neg.to_csv(OUT / "p2a_negative_yields.csv")
    print("(a) descriptives\n", desc.round(3), "\n", neg)

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for c, col in zip(["2Yr", "5Yr", "10Yr", "40Yr"], COLORS):
        ax.plot(gilts.index, gilts[c], color=col, label=c, lw=1.2)
    ax.axhline(0, color="#52514e", lw=0.8)
    for date, txt in [("2020-03-11", "COVID-19: BoE cuts to 0.1%,\nQE restarts"),
                      ("2021-12-16", "first BoE hike"),
                      ("2022-09-23", "mini-budget")]:
        ax.axvline(pd.Timestamp(date), color="#8a8984", lw=0.8, ls="--")
        ax.text(pd.Timestamp(date), 4.75, txt, fontsize=7, color="#52514e", ha="right", va="top",
                rotation=0)
    ax.axvspan(pd.Timestamp("2020-05-21"), pd.Timestamp("2021-05-26"), color="#e6e5e0", alpha=.6, lw=0)
    ax.text(pd.Timestamp("2020-11-15"), -0.45, "negative short yields", fontsize=7,
            ha="center", color="#52514e")
    ax.set(ylabel="yield (%)", title="UK gilt yields, June 2018 - October 2022", ylim=(-0.6, 4.9))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 0.88))
    fig.savefig(OUT / "fig_p2a_yields.png"); plt.close(fig)

    mats = np.array([0.75, 10/12, 11/12, 1, 2, 3, 5, 7, 10, 13, 17, 21, 26, 32, 40])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    periods = {"full-sample mean": gilts,
               "pre-COVID (to Feb 2020)": gilts[:"2020-02-29"],
               "ZLB (Apr 2020 - Nov 2021)": gilts["2020-04-01":"2021-11-30"],
               "Oct 2022": gilts["2022-10-01":]}
    for (lab, g), col in zip(periods.items(), ["#0b0b0b"] + COLORS[:3]):
        ax.plot(mats, g.mean(), marker="o", ms=3.5, color=col, label=lab,
                lw=2 if lab.startswith("full") else 1.2)
    ax.set(xlabel="maturity (years)", ylabel="average yield (%)",
           title="Average yield curve, full sample and sub-periods")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.62), fontsize=7.5)
    fig.savefig(OUT / "fig_p2a_curve.png"); plt.close(fig)
    curve = pd.DataFrame({k: v.mean() for k, v in periods.items()})
    curve.round(3).to_csv(OUT / "p2a_curve.csv")
    print(curve.round(3))

    # ---------------- (b) ADF on 2Yr and 10Yr, p = 5 ----------------
    from statsmodels.tsa.stattools import adfuller    # cross-check only
    from statsmodels.tsa.adfvalues import mackinnoncrit
    tab_b = pd.DataFrame({c: Q2_ADF(gilts[c], 5) for c in ["2Yr", "10Yr"]}).T
    tab_b_adj = pd.DataFrame({c: Q2_ADF(gilts[c], 5, adjusted=True) for c in ["2Yr", "10Yr"]}).T
    for c in ["2Yr", "10Yr"]:
        for reg, col in (("c", "t (const)"), ("ct", "t (trend)")):
            sm = adfuller(gilts[c], maxlag=5, autolag=None, regression=reg, result_object=False)[0]
            assert np.isclose(sm, tab_b.loc[c, col]), (c, reg)   # our OLS = statsmodels
    T_b = len(y10) - 1 - 5
    # textbook critical values: MacKinnon (2010) response surface for t at this T
    # (mackinnoncrit returns 1%, 5%, 10%), and the asymptotic normalized-bias values of
    # Fuller (1976) / Hamilton (1994, table B.5, cases 2 and 4)
    textbook = pd.DataFrame({
        "t (const)": mackinnoncrit(1, "c", T_b)[:3], "t (trend)": mackinnoncrit(1, "ct", T_b)[:3],
        "delta (const)": [-20.7, -14.1, -11.3], "delta (trend)": [-29.5, -21.8, -18.3]},
        index=["99%", "95%", "90%"]).iloc[::-1]
    # robustness: (1) before the 2022 tightening cycle, (2) on first differences (is it I(2)?)
    tab_b_extra = pd.DataFrame({
        "2Yr, to Dec 2021": Q2_ADF(y2[:"2021-12-31"], 5), "10Yr, to Dec 2021": Q2_ADF(y10[:"2021-12-31"], 5),
        "Δ2Yr": Q2_ADF(y2.diff().dropna(), 5), "Δ10Yr": Q2_ADF(y10.diff().dropna(), 5)}).T
    tab_b_extra.round(3).to_csv(OUT / "p2b_adf_robustness.csv")
    print(tab_b_extra.round(2))
    tab_b.round(3).to_csv(OUT / "p2b_adf.csv"); tab_b_adj.round(3).to_csv(OUT / "p2b_adf_adjusted.csv")
    textbook.round(3).to_csv(OUT / "p2b_textbook_cv.csv")
    print("(b)\n", tab_b.round(3), "\n adjusted delta\n", tab_b_adj.round(3), "\n", textbook.round(2))

    # ---------------- (c) bootstrap CVs from the 10Yr, p=5, T=400, N=500 ----------------
    cv_c = Q2_CVSim(y10, N=500, T=400, p=5, rng=np.random.default_rng(SEED))
    # precision check: same design with N = 20,000, and the Gaussian (textbook) design
    big = 20_000
    draws_uk = simulate_null(y10, big, 400, 5, np.random.default_rng(SEED + 1))
    draws_gauss = simulate_null(y10, big, 400, 5, np.random.default_rng(SEED + 2), gaussian=True)
    pct = lambda a: pd.DataFrame(np.percentile(a, [10, 5, 1], axis=0),
                                 index=["90%", "95%", "99%"], columns=STAT_NAMES)
    cv_c_big, cv_gauss = pct(draws_uk), pct(draws_gauss)
    # Monte Carlo standard error of each N=500 critical value: spread over 200 replications
    # of the N=500 experiment, obtained by resampling 500 draws from the 20,000.
    reps = np.stack([pct(draws_uk[np.random.default_rng(k).choice(big, 500)]).to_numpy()
                     for k in range(200)])
    mcse = pd.DataFrame(reps.std(axis=0), index=cv_c.index, columns=STAT_NAMES)
    for name, t in [("p2c_cv_N500", cv_c), ("p2c_cv_N20000", cv_c_big),
                    ("p2c_cv_gaussian", cv_gauss), ("p2c_mcse_N500", mcse)]:
        t.round(3).to_csv(OUT / f"{name}.csv")
    print("(c) N=500\n", cv_c.round(3), "\n N=20000\n", cv_c_big.round(3),
          "\n Gaussian\n", cv_gauss.round(3), "\n MC s.e.\n", mcse.round(3))

    # ---------------- (d) S&P 500, last 400 obs, p = 3 ----------------
    spx400 = lspx.iloc[-400:]
    cv_d = Q2_CVSim(spx400, N=500, T=400, p=3, rng=np.random.default_rng(SEED))
    draws_spx = simulate_null(spx400, big, 400, 3, np.random.default_rng(SEED + 3))
    draws_spx_dm = simulate_null(spx400, big, 400, 3, np.random.default_rng(SEED + 3), demean=True)
    draws_uk_p3 = simulate_null(y10, big, 400, 3, np.random.default_rng(SEED + 4))
    cv_d_big, cv_d_dm, cv_uk_p3 = pct(draws_spx), pct(draws_spx_dm), pct(draws_uk_p3)
    adf_spx = Q2_ADF(spx400, 3)
    # drift-to-noise ratio mu*sqrt(T)/sigma: how visible the drift is over T = 400
    dsp, d10 = np.diff(spx400.to_numpy()), np.diff(y10.to_numpy())
    drift = pd.DataFrame({
        "mean of increments": [d10.mean(), dsp.mean()], "sd of increments": [d10.std(), dsp.std()],
        "drift/noise over T=400": [d10.mean() * 20 / d10.std(), dsp.mean() * 20 / dsp.std()]},
        index=["UK 10Yr (full sample)", "log S&P 500 (last 400)"])
    # bootstrap p-values of the observed S&P statistics (from the N = 20,000 draws)
    pval = pd.Series((draws_spx <= adf_spx.to_numpy()).mean(axis=0), index=STAT_NAMES)
    for name, t in [("p2d_cv_N500", cv_d), ("p2d_cv_N20000", cv_d_big),
                    ("p2d_cv_demeaned", cv_d_dm), ("p2d_cv_uk_p3", cv_uk_p3), ("p2d_drift", drift)]:
        t.round(4).to_csv(OUT / f"{name}.csv")
    pd.DataFrame({"statistic": adf_spx, "95% CV (N=500)": cv_d.loc["95%"],
                  "bootstrap p-value": pval}).round(3).to_csv(OUT / "p2d_spx_adf.csv")
    print("(d) N=500\n", cv_d.round(3), "\n N=20000\n", cv_d_big.round(3), "\n demeaned\n",
          cv_d_dm.round(3), "\n UK with p=3\n", cv_uk_p3.round(3), "\n", drift.round(4),
          "\n S&P ADF\n", adf_spx.round(3), "\n p-values\n", pval.round(3),
          "\n S&P window", spx400.index[0].date(), spx400.index[-1].date())

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3), sharey=False)
    for k, ax in enumerate(axes):
        for arr, lab, col in [(draws_gauss[:, k], "Gaussian RW", "#8a8984"),
                              (draws_uk[:, k], "UK 10Yr bootstrap", COLORS[0]),
                              (draws_spx[:, k], "S&P 500 bootstrap", COLORS[1])]:
            ax.hist(arr, bins=80, density=True, histtype="step", color=col, lw=1.2, label=lab)
        ax.axvline(adf_spx.iloc[k], color="#0b0b0b", lw=0.9, ls=":")
        ax.set_title(STAT_NAMES[k].replace("delta", "δ"), fontsize=8.5)
        ax.set_yticks([]); ax.grid(False)
        if k >= 2: ax.set_xlim(-45 if k == 2 else -60, 8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    fig.suptitle("Null distributions of the four ADF statistics (T = 400, 20,000 walks each); "
                 "dotted: observed S&P 500 statistic", fontsize=8.5, y=1.04)
    fig.savefig(OUT / "fig_p2d_null_dists.png"); plt.close(fig)

    # ---------------- (e) rolling ADF, 10Yr, p = 5, N = 250 ----------------
    roll_uk, path_uk = Q2_RollingADF(y10, p=5, N=250, return_path=True)
    min_null_uk = rolling_min_null(y10, 5, 250, reps=1000, rng=np.random.default_rng(SEED + 5))
    cv_min_uk = pct(min_null_uk)
    share_rej_uk = (path_uk < cv_c_big.loc["95%"]).mean()
    roll_uk.to_csv(OUT / "p2e_rolling.csv"); cv_min_uk.round(3).to_csv(OUT / "p2e_cv_min.csv")
    share_rej_uk.round(3).to_csv(OUT / "p2e_share_rejecting.csv")
    print("(e)\n", roll_uk, "\n CV of the min\n", cv_min_uk.round(2), "\n share of windows < 5% CV\n",
          share_rej_uk.round(3))

    # ---------------- (f) rolling ADF, log S&P 500 last 1,500 obs, p = 3, N = 250 -----------
    spx1500 = lspx.iloc[-1500:]
    roll_spx, path_spx = Q2_RollingADF(spx1500, p=3, N=250, return_path=True)
    min_null_spx = rolling_min_null(spx1500, 3, 250, reps=1000, rng=np.random.default_rng(SEED + 6))
    cv_min_spx = pct(min_null_spx)
    cv_spx_250 = pct(simulate_null(spx1500, 5000, 250, 3, np.random.default_rng(SEED + 7)))
    share_rej_spx = (path_spx < cv_spx_250.loc["95%"]).mean()
    roll_spx.to_csv(OUT / "p2f_rolling.csv"); cv_min_spx.round(3).to_csv(OUT / "p2f_cv_min.csv")
    cv_spx_250.round(3).to_csv(OUT / "p2f_cv_window.csv")
    print("(f) sample", spx1500.index[0].date(), spx1500.index[-1].date(), "\n", roll_spx,
          "\n CV of the min\n", cv_min_spx.round(2), "\n single-window CV (T=250)\n", cv_spx_250.round(2),
          "\n share of windows < 5% CV\n", share_rej_spx.round(3))

    # rolling figures: level on top, t-statistics and deltas below, window END dates on x
    for tag, series, path, roll, cvw, title in [
            ("e", y10, path_uk, roll_uk, cv_c_big, "UK 10-year gilt yield (%)"),
            ("f", spx1500, path_spx, roll_spx, cv_spx_250, "log S&P 500")]:
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.1, 1, 1]})
        axes[0].plot(series.index, series, color="#0b0b0b", lw=1)
        axes[0].set_ylabel(title.split(" (")[0] if tag == "f" else "yield (%)")
        axes[0].set_title(f"Rolling ADF, windows of 250 days: {title}")
        for ax, cols in zip(axes[1:], [STAT_NAMES[:2], STAT_NAMES[2:]]):
            for c, col in zip(cols, COLORS[:2]):
                k = STAT_NAMES.index(c)
                ax.plot(path.index, path[c], color=col, lw=1.1, label=c.replace("delta", "δ"))
                ax.axhline(cvw.loc["95%", c], color=col, lw=0.8, ls="--")
            ax.legend(loc="lower left", ncol=2, fontsize=7.5)
        axes[1].set_ylabel("t-statistic"); axes[2].set_ylabel(r"$\delta = T\hat\beta$")
        # mark the window that produced each statistic's minimum as a bar above the series
        lo, hi = series.min(), series.max()
        for k, (c, col) in enumerate(zip(STAT_NAMES, COLORS[:2] * 2)):
            yk = hi + (0.10 + 0.09 * k) * (hi - lo)
            s0, e0 = pd.Timestamp(roll.loc["window start", c]), pd.Timestamp(roll.loc["window end", c])
            axes[0].plot([s0, e0], [yk, yk], color=col, lw=2.5, solid_capstyle="butt",
                         ls="-" if k < 2 else (0, (2, 1)))
            axes[0].text(e0, yk, "  min " + c.replace("delta", "δ"), fontsize=6.5, va="center",
                         color="#52514e")
        axes[0].set_ylim(lo - 0.05 * (hi - lo), hi + 0.5 * (hi - lo))
        axes[2].set_xlabel("end date of the 250-day window   (dashed: single-window 5% critical values)")
        fig.savefig(OUT / f"fig_p2{tag}_rolling.png"); plt.close(fig)


if __name__ == "__main__":
    main()
