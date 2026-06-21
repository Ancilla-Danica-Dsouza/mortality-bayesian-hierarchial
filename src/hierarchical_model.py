"""
hierarchical_model.py
======================
Hierarchical Bayesian extension of Lee-Carter, implemented in NumPyro.

Model (linear covariate link, random-walk time prior -- default variant):

    ln(m_x,t,c) = a_{x,c} + b_{x,c} * k_{t,c} + gamma^T S_{c,t} + eps_x,t,c

    Hierarchical priors (partial pooling across countries c):
        a_{x,c} ~ Normal(mu_a[x], tau_a)
        b_{x,c} ~ Normal(mu_b[x], tau_b)
        mu_a[x], mu_b[x]  ~ Normal(0, 5)      (global age-profile means)
        tau_a, tau_b      ~ HalfCauchy(2)     (pooling strength, learned)

    Time dynamics for k_{t,c}, two variants (--time-prior flag):
        randomwalk : k_t = k_(t-1) + drift_c + noise
        gp         : k_t ~ GP(0, kernel(t, t'))   (squared-exponential kernel)

    Covariate link, two variants (--link flag):
        linear : gamma^T S_{c,t}
        neural : f_theta(S_{c,t})   (small MLP, weights given weakly-informative priors)

Usage:
    python src/hierarchical_model.py --link linear --time-prior randomwalk
    python src/hierarchical_model.py --link neural  --time-prior gp
"""

import argparse
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


def model_linear_randomwalk(
    age_idx: jnp.ndarray,
    time_idx: jnp.ndarray,
    country_idx: jnp.ndarray,
    covariates: jnp.ndarray,
    n_ages: int,
    n_years: int,
    n_countries: int,
    log_mortality_obs: jnp.ndarray = None,
):
    """
    NumPyro model: linear covariate link + random-walk-with-drift time index.

    Parameters
    ----------
    age_idx, time_idx, country_idx : integer index arrays, one entry per observation
    covariates : array, shape (n_obs, n_covariates)
    log_mortality_obs : observed ln(m_x,t,c), or None when sampling/forecasting

    TODO:
        1. Define global hyperpriors mu_a, mu_b (shape n_ages) and tau_a, tau_b (scalars)
        2. Sample a_{x,c}, b_{x,c} with numpyro.plate over countries, conditioned on
           mu_a[x], mu_b[x], tau_a, tau_b
        3. Sample k_{t,c} as a random walk: k_0 ~ Normal(0,1); k_t = k_(t-1) + drift_c + noise
        4. Sample gamma (covariate coefficients) ~ Normal(0, 1) per covariate
        5. Compute mean = a_{x,c} + b_{x,c} * k_{t,c} + gamma @ covariates
        6. Sample observation noise sigma ~ HalfCauchy(1)
        7. numpyro.sample("obs", dist.Normal(mean, sigma), obs=log_mortality_obs)
    """
    raise NotImplementedError


def model_neural_link(
    age_idx: jnp.ndarray,
    time_idx: jnp.ndarray,
    country_idx: jnp.ndarray,
    covariates: jnp.ndarray,
    n_ages: int,
    n_years: int,
    n_countries: int,
    hidden_dim: int = 8,
    log_mortality_obs: jnp.ndarray = None,
):
    """
    Variant of the hierarchical model where the covariate link is a small MLP
    f_theta(S_{c,t}) instead of a linear term gamma^T S_{c,t}.

    TODO:
        - Same hierarchical structure for a_{x,c}, b_{x,c}, k_{t,c} as the linear model
        - Replace linear term with a 1-hidden-layer MLP:
              h = tanh(W1 @ covariates + b1)
              covariate_effect = W2 @ h + b2
          with weakly-informative Normal(0, 1) priors on W1, b1, W2, b2
    """
    raise NotImplementedError


def model_gp_time_prior(*args, **kwargs):
    """
    Variant using a Gaussian Process prior over k_{t,c} instead of a random walk.

    TODO:
        - Define a squared-exponential kernel function over time indices
        - Sample k_{t,c} jointly via a multivariate normal with that covariance
        - Combine with either the linear or neural covariate link
    """
    raise NotImplementedError


def run_mcmc(model_fn, *model_args, num_warmup=1000, num_samples=2000, **model_kwargs):
    """
    Run NUTS MCMC for a given model function. Returns the MCMC object
    (use .get_samples() to extract posterior draws).

    TODO: implement NUTS kernel setup and mcmc.run() call.
    """
    raise NotImplementedError


def run_svi(model_fn, *model_args, num_steps=5000, **model_kwargs):
    """
    Run stochastic variational inference (faster, approximate) for scaling
    to the full multi-country panel.

    TODO: implement AutoNormal guide + SVI training loop, return learned params.
    """
    raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", choices=["linear", "neural"], default="linear")
    parser.add_argument("--time-prior", choices=["randomwalk", "gp"], default="randomwalk")
    parser.add_argument("--inference", choices=["mcmc", "svi"], default="mcmc")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: load data/processed/panel.csv, build index arrays + covariate matrix,
    # select model variant based on args, run inference, save posterior samples.
    print(f"Hierarchical model scaffold ready (link={args.link}, time_prior={args.time_prior}).")


if __name__ == "__main__":
    main()