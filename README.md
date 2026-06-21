# Hierarchical Bayesian Mortality Forecasting with Socioeconomic Covariates

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPyro](https://img.shields.io/badge/built%20with-NumPyro-orange)](https://num.pyro.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20progress-brightgreen)](#)

A hierarchical Bayesian extension of the Lee–Carter mortality model that incorporates
socioeconomic covariates (GDP per capita, healthcare expenditure, education) to improve
out-of-sample life-expectancy forecasts, with **distribution-free posterior predictive
risk bounds** obtained via conformal calibration.

> **TL;DR** — Classic mortality models (Lee–Carter) extrapolate age- and time-specific
> death rates but ignore *why* mortality differs across countries. This project pools
> information across countries via a hierarchical prior, lets socioeconomic indicators
> explain part of that variation, and wraps the resulting posterior in a calibration
> layer that gives a provable coverage guarantee on the forecast intervals — something
> standard Bayesian credible intervals don't give you for free.

---

## Table of Contents

- [Motivation](#motivation)
- [Model](#model)
- [Posterior Predictive Risk Bound](#posterior-predictive-risk-bound)
- [Data](#data)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Roadmap](#roadmap)
- [References](#references)

---

## Motivation

The **Lee–Carter model** (Lee & Carter, 1992) is the workhorse of actuarial mortality
forecasting:

$$
\ln(m_{x,t}) = a_x + b_x k_t + \varepsilon_{x,t}
$$

where $m_{x,t}$ is the central death rate at age $x$ in year $t$, $a_x$ is the average
log-mortality age profile, $b_x$ captures how sensitive each age group is to the
time-varying mortality index $k_t$, and $k_t$ is typically modeled as a random walk
with drift.

This is elegant but has two limitations this project addresses:

1. **No covariates.** Two countries with identical past mortality trajectories but
   very different GDP growth get the same forecast.
2. **No pooling across populations.** Small or noisy national datasets are fit in
   isolation, wasting information that similar countries could share.
3. **No distribution-free uncertainty guarantee.** Bayesian credible intervals are
   only as good as the model is correctly specified — they don't carry a frequentist
   coverage guarantee.

This project tackles all three.

---

## Model

### Baseline
A from-scratch SVD-based replication of Lee–Carter (no external mortality-modeling
package), used as the benchmark every extension is measured against.

### Hierarchical Bayesian extension

For country $c$, age $x$, year $t$:

$$
\ln(m_{x,t,c}) = a_{x,c} + b_{x,c}\, k_{t,c} + \gamma^\top S_{c,t} + \varepsilon_{x,t,c}
$$

with **partial pooling** across countries:

$$
a_{x,c} \sim \mathcal{N}(\mu_{a,x}, \tau_a^2), \qquad
b_{x,c} \sim \mathcal{N}(\mu_{b,x}, \tau_b^2)
$$

$S_{c,t}$ is a vector of socioeconomic covariates (GDP per capita, health expenditure
% of GDP, mean years of schooling). Countries with sparse data shrink toward the
global age profile $\mu_{a,x}, \mu_{b,x}$; data-rich countries are barely shrunk at all.
This is a standard bias–variance trade-off made explicit through the hierarchy's
variance parameters $\tau_a, \tau_b$.

Two variants are implemented for the covariate link $\gamma^\top S_{c,t}$:

- **Linear** — interpretable, fewer parameters, easy to validate against domain priors.
- **Neural** — a small feed-forward network $f_\theta(S_{c,t})$ replacing the linear
  term, allowing nonlinear interactions between socioeconomic indicators (e.g.
  diminishing returns to healthcare spending at high GDP).

Inference is performed with **NumPyro** using both MCMC (NUTS) for ground-truth
posteriors on smaller runs and stochastic variational inference (SVI) for scaling to
the full multi-country panel.

### Time dynamics

Two priors are compared for the latent mortality index $k_{t,c}$:

- **Random walk with drift** (classical Lee–Carter assumption)
- **Gaussian Process prior**, allowing smoother, nonlinear trend extrapolation with a
  principled kernel-based uncertainty band

---

## Posterior Predictive Risk Bound

Bayesian credible intervals are only calibrated if the model is well-specified — a
strong assumption here. To get a guarantee that does **not** depend on model
correctness, posterior predictive draws are passed through **split conformal
calibration**:

1. Fit the hierarchical model on a training split.
2. Compute nonconformity scores on a held-out calibration split (e.g. absolute
   posterior-mean residuals, scaled by posterior predictive std).
3. Use the empirical quantile of these scores to widen/narrow the Bayesian interval
   so that it achieves **finite-sample, distribution-free coverage**:

$$
\mathbb{P}\left(m_{x,t+h,c} \in \hat{C}(x,t+h,c)\right) \geq 1 - \alpha
$$

This holds regardless of whether the hierarchical model is correctly specified —
only exchangeability of calibration/test residuals is required. The full derivation,
including the exchangeability argument adapted to the panel/longitudinal setting, is
in [`docs/derivations.pdf`](docs/derivations.pdf).

---

## Data

| Source | Content | Link |
|---|---|---|
| Human Mortality Database (HMD) | Age/year/country mortality and exposure tables | https://www.mortality.org |
| World Bank Open Data | GDP per capita, health expenditure, education indicators | https://data.worldbank.org |

Raw files are not committed to the repo (see `.gitignore`); `src/data_prep.py`
downloads and harmonizes them into `data/processed/`.

---

## Repository Structure

```
mortality-bayesian-hierarchical/
├── data/
│   ├── raw/                  # untouched downloads (gitignored)
│   └── processed/            # cleaned, merged panel data
├── notebooks/                # EDA, posterior diagnostics, scratch work
├── src/
│   ├── data_prep.py          # HMD + World Bank ingestion and merging
│   ├── leecarter.py          # from-scratch SVD baseline
│   ├── hierarchical_model.py # NumPyro hierarchical Bayesian model
│   ├── conformal.py          # split conformal calibration wrapper
│   └── metrics.py            # RMSE, interval coverage, CRPS, etc.
├── docs/
│   ├── derivations.pdf       # full math write-up of the risk bound proof
│   └── model_card.md         # assumptions, limitations, intended use
├── tests/                    # convergence diagnostics as unit tests (R-hat, ESS)
├── results/
│   └── figures/              # generated plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/mortality-bayesian-hierarchical.git
cd mortality-bayesian-hierarchical
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Prepare data
python src/data_prep.py

# 2. Fit baseline Lee-Carter
python src/leecarter.py

# 3. Fit hierarchical Bayesian model
python src/hierarchical_model.py --link linear --time-prior randomwalk

# 4. Apply conformal calibration and evaluate
python src/conformal.py
```

---

## Results

*(filled in as experiments complete — Week 7–8 of the project plan)*

| Model | Forecast RMSE | 90% Interval Coverage | Notes |
|---|---|---|---|
| Lee–Carter (baseline) | — | — | SVD, no covariates |
| Hierarchical, linear link, RW prior | — | — | |
| Hierarchical, linear link, GP prior | — | — | |
| Hierarchical, neural link, RW prior | — | — | |
| + Conformal calibration | — | — | distribution-free coverage |

---

## Roadmap

- [x] Repository scaffolding
- [ ] Data pipeline (HMD + World Bank merge)
- [ ] Lee–Carter baseline (SVD, from scratch)
- [ ] Hierarchical Bayesian model (NumPyro, linear link)
- [ ] Neural covariate link variant
- [ ] Gaussian Process time-prior variant
- [ ] Conformal calibration layer + coverage proof
- [ ] Ablation study (pooling vs. none, linear vs. neural, RW vs. GP)
- [ ] Final write-up (`docs/derivations.pdf`) and results dashboard

## References

- Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality.
  *Journal of the American Statistical Association*, 87(419), 659–671.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.).
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*
  (conformal prediction).
- Human Mortality Database. University of California, Berkeley, and Max Planck
  Institute for Demographic Research. https://www.mortality.org