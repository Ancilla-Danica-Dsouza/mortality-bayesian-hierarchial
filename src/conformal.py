"""
conformal.py
============
Wraps Bayesian posterior predictive draws in split conformal calibration to
obtain finite-sample, distribution-free coverage guarantees on forecast
intervals -- independent of whether the hierarchical model is correctly
specified.

Procedure (split conformal, adapted to the longitudinal/panel setting):
    1. Split data into proper-train / calibration / test sets
       (calibration set = held-out time period or held-out countries).
    2. Fit the hierarchical model on proper-train only.
    3. For each calibration point, compute a nonconformity score:
           score_i = |y_i - posterior_mean_i| / posterior_std_i
       (a normalized residual -- accounts for heteroskedastic uncertainty
       across ages/countries).
    4. Take the empirical (1 - alpha)-quantile of calibration scores: q_hat.
    5. For test points, construct the conformal interval:
           C(x) = [posterior_mean(x) - q_hat * posterior_std(x),
                    posterior_mean(x) + q_hat * posterior_std(x)]
    6. This interval satisfies: P(y in C(x)) >= 1 - alpha,
       under the sole assumption that calibration and test nonconformity
       scores are exchangeable.

See docs/derivations.pdf for the full proof, including the exchangeability
argument adapted from i.i.d. conformal prediction to the panel-data setting
(exchangeability across countries within a fixed time period).
"""

import numpy as np


def compute_nonconformity_scores(
    y_true: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized absolute-residual nonconformity scores on the
    calibration set.

    score_i = |y_i - posterior_mean_i| / posterior_std_i

    TODO: implement; guard against posterior_std == 0 with a small epsilon.
    """
    raise NotImplementedError


def compute_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the conformal quantile q_hat: the ceil((n+1)(1-alpha))/n -th
    empirical quantile of the calibration scores (finite-sample correction,
    not just the naive (1-alpha) quantile).

    TODO: implement using the standard split-conformal quantile correction.
    """
    raise NotImplementedError


def build_conformal_intervals(
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build calibrated [lower, upper] intervals for test points.

    Returns
    -------
    lower, upper : np.ndarray
        Conformal interval bounds, same shape as posterior_mean.

    TODO: implement interval construction:
        lower = posterior_mean - q_hat * posterior_std
        upper = posterior_mean + q_hat * posterior_std
    """
    raise NotImplementedError


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Compute the fraction of true values falling inside [lower, upper] --
    used to empirically verify the theoretical coverage guarantee on a
    held-out test set.

    TODO: implement as mean((y_true >= lower) & (y_true <= upper)).
    """
    raise NotImplementedError


def main():
    # TODO: load posterior predictive samples (from hierarchical_model.py output),
    # split into calibration/test, run the pipeline above, report empirical coverage
    # vs. target (1 - alpha) across several alpha values (e.g. 0.1, 0.2).
    print("Conformal calibration scaffold ready. Implement TODOs to run end-to-end.")


if __name__ == "__main__":
    main()