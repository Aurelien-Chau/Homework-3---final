# MATH 60231 Financial Econometrics: Homework 3

HEC Montréal, Fall 2026. Antoine Djalane, Aurélien Chaubell, Arslan Berrah.

The write-up (HW3 report, PDF) is submitted separately. This repository holds the code and data that produce every number, table and figure in it.

## Running the code

The only requirement is [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.13 and the exact package versions pinned in `uv.lock`.

```bash
uv sync                               # create .venv with the locked environment
uv run python hwk3_problem1.py        # Problem 1 (PJM electricity prices), ~30 s
uv run python hwk3_problem2.py        # Problem 2 (UK gilts, S&P 500),     ~2.5 min
```

Each script prints its main results and writes:

- tables to `output/*.csv`;
- figures to `output/fig_*.png`.

The `output/` folder is created on the first run. Results are reproducible: every simulation uses NumPy's PCG64 generator with seed 60231, so the scripts regenerate exactly the numbers in the report.

## Contents

| Path | Contents |
|---|---|
| `hwk3_problem1.py` | `Q1_ARForecast`, `Q1_AREstimate`, `Q1_SimPath`, `Q1_SimBands`; rolling hourly AR(24) vs daily AR(6) and the DM test (`Q2_DM` reused unchanged from Homework 2) |
| `hwk3_problem2.py` | `Q2_ADF`, `Q2_CVSim`, `Q2_RollingADF`; descriptive statistics, bootstrap critical values, rolling ADF and the bootstrap of the rolling minimum |
| `data/` | `PJM.csv`, `UK_bonds.parquet`, `S&P 500 Historical Data.csv`, as provided with the assignment |
| `output/` | Created by the scripts: every table (`p1*.csv`, `p2*.csv`) and figure (`fig_p1*.png`, `fig_p2*.png`) |
| `pyproject.toml`, `uv.lock`, `.python-version` | The uv environment |

Problem 3 (true/false) has no code. Its answers are in section 3 of the report.

## Implementation notes

- **Required functions.** The functions named in the assignment keep their names and signatures. Optional keyword arguments, such as `rng` and `adjusted`, have defaults that reproduce the required behaviour.
- **Vectorization.** ADF regressions are vectorized: `Q2_CVSim` and `Q2_RollingADF` solve the OLS problems of all walks or windows in a single batched call. The results are identical to calling `Q2_ADF` one series at a time.
- **Cross-checks.** Our OLS estimates match `statsmodels` exactly. The script asserts that our ADF t-statistics equal `statsmodels.tsa.stattools.adfuller`.
- **Problem 1 sample.** Problem 1 uses the last 8,760 hours of `PJM.csv`, from 2017-12-25 00:00 to 2018-12-24 23:00. Problem 1(e) evaluates 65 target days within that sample, plus a 365-day robustness check whose estimation windows reach back before it.
