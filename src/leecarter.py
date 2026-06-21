"""
leecarter.py
============
From-scratch implementation of the Lee-Carter mortality model (Lee & Carter, 1992),
used as the baseline against which the hierarchical Bayesian extension is measured.

Model:
    ln(m_x,t) = a_x + b_x * k_t + epsilon_x,t

Estimation (classical, SVD-based):
    1. a_x = mean over t of ln(m_x,t)
    2. Center the log-mortality matrix: Z_x,t = ln(m_x,t) - a_x
    3. Take rank-1 SVD of Z: Z ≈ s1 * u1 * v1^T
    4. b_x = u1 (normalized so sum(b_x) = 1)
       k_t = s1 * v1 (rescaled accordingly so sum(k_t) = 0)
    5. Forecast k_t forward using a random walk with drift:
           k_t = k_(t-1) + drift + noise

Run directly:
    python src/leecarter.py
"""

import numpy as np
import pandas as pd


def fit_lee_carter(log_mortality_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the classical Lee-Carter model via SVD.

    Parameters
    ----------
    log_mortality_matrix : np.ndarray, shape (n_ages, n_years)
        ln(m_x,t) for a single country/population.

    Returns
    -------
    a_x : np.ndarray, shape (n_ages,)
    b_x : np.ndarray, shape (n_ages,)   -- normalized so sum(b_x) = 1
    k_t : np.ndarray, shape (n_years,)  -- normalized so sum(k_t) = 0

    TODO: implement SVD decomposition + normalization constraints.
    """
    raise NotImplementedError


def fit_drift_rw(k_t: np.ndarray) -> tuple[float, float]:
    """
    Fit a random walk with drift to the time index k_t:
        k_t = k_(t-1) + drift + noise,  noise ~ N(0, sigma^2)

    Returns
    -------
    drift : float
    sigma : float (residual standard deviation)

    TODO: implement via simple first-differencing (drift = mean diff,
    sigma = std of differences).
    """
    raise NotImplementedError


def forecast_lee_carter(
    a_x: np.ndarray,
    b_x: np.ndarray,
    k_t_last: float,
    drift: float,
    sigma: float,
    horizon: int,
    n_sims: int = 1000,
) -> np.ndarray:
    """
    Forecast log-mortality rates `horizon` years ahead, with simulation-based
    uncertainty from the random walk noise.

    Returns
    -------
    forecasts : np.ndarray, shape (n_sims, n_ages, horizon)
        Simulated forecasted ln(m_x,t) paths, for computing point forecasts
        and prediction intervals.

    TODO: implement Monte Carlo simulation of k_t forward, then reconstruct
    ln(m_x,t) = a_x + b_x * k_t for each simulated path.
    """
    raise NotImplementedError


def evaluate_forecast(true_log_mortality: np.ndarray, forecasts: np.ndarray) -> dict:
    """
    Compute evaluation metrics comparing forecasts to held-out true values.

    Returns dict with keys: 'rmse', 'mae', 'coverage_90'.

    TODO: implement metric calculations (see also src/metrics.py for shared utilities).
    """
    raise NotImplementedError


def main():
    # TODO: load data/processed/panel.csv, pivot to (age x year) matrix per country,
    # fit, forecast on a held-out time window, and report baseline metrics.
    print("Lee-Carter baseline scaffold ready. Implement TODOs to run end-to-end.")


if __name__ == "__main__":
    main()