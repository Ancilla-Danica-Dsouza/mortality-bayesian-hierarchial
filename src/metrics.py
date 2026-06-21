"""
metrics.py
==========
Shared evaluation metrics used across leecarter.py, hierarchical_model.py,
and conformal.py, so every model is scored identically.
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error. TODO: implement."""
    raise NotImplementedError


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. TODO: implement."""
    raise NotImplementedError


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values within [lower, upper]. TODO: implement."""
    raise NotImplementedError


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width of prediction intervals (sharpness metric). TODO: implement."""
    raise NotImplementedError


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictive distributions --
    a proper scoring rule combining calibration and sharpness in one number.

    Closed-form CRPS for N(mean, std^2):
        CRPS = std * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
        where z = (y_true - mean) / std

    TODO: implement using scipy.stats.norm for Phi (cdf) and phi (pdf).
    """
    raise NotImplementedError


def r_hat_summary(mcmc_samples: dict) -> dict:
    """
    Wrapper around ArviZ's r_hat diagnostic for MCMC convergence checking.
    Returns a dict of {param_name: r_hat_value}.

    TODO: implement using arviz.rhat() on an arviz.InferenceData object.
    """
    raise NotImplementedError