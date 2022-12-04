from functools import partial
from collections import defaultdict, namedtuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit

from ..utils import softplus, split_params


# =============================================================================
# Standard Gaussian Process Regression
# =============================================================================


def _gpr_log_marginal_likelihood(params, x, y, kernel, return_negative=False):
    """
    Computes the log marginal likelihood for standard Gaussian Process Regression.
    Arguments
    ---------
    params  : dict
            Dictionary of parameters. Should have a 'kernel_params' keyword
            to specify kernel parameters (a ictionary) and a 'sigma' keyword
            to specify the noise.
    x       : jnp.ndarray, (M, F)
            Input matrix of M samples and F features
    y       : jnp.ndarray, (M, 1)
            Target matrix of M samples and 1 target
    kernel  : callable
            Kernel function
    Returns
    -------
    lml     : jnp.ndarray, ()
            Log marginal likelihood
    """

    kernel_params, sigma = split_params(params)
    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
    sigma = softplus(sigma)

    y_mean = jnp.mean(y)
    y = y - y_mean
    m = y.shape[0]

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(m)
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)
    L_m = jsp.linalg.cholesky(C_mm, lower=True)
    mll = (
        -0.5 * jnp.squeeze(jnp.dot(y.T, c))
        - jnp.sum(jnp.log(jnp.diag(L_m)))
        - m * 0.5 * jnp.log(2.0 * jnp.pi)
    )
    if return_negative:
        return -mll
    return mll


def _gpr_fit(params, x, y, kernel):
    """
    Fits a Gaussian Process Regression model.
    Arguments
    ---------
    params  : dict
            Dictionary of parameters. Should have a 'kernel_params' keyword
            to specify kernel parameters (a ictionary) and a 'sigma' keyword
            to specify the noise.
    x       : jnp.ndarray, (M, F)
            Input matrix of M samples and F features
    y       : jnp.ndarray, (M, 1)
            Target matrix of M samples and 1 target
    kernel  : callable
            Kernel function
    Returns
    -------
    c       : jnp.ndarray, (M, 1)
            Dual coefficients
    y_mean  : jnp.ndarray, ()
            Target mean
    """

    kernel_params, sigma = split_params(params)
    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
    sigma = softplus(sigma)

    y_mean = jnp.mean(y)
    y = y - y_mean

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(x.shape[0])
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)

    return c, y_mean


def _gpr_predict(
    params,
    x_train,
    x,
    c,
    y_mean,
    kernel,
    full_covariance=False,
):
    """
    Predict with a Gaussian Process Regression model.
    Arguments
    ---------
    params          : dict
                    Dictionary of parameters. Should have a 'kernel_params' keyword
                    to specify kernel parameters (a ictionary) and a 'sigma' keyword
                    to specify the noise.
    x               : jnp.ndarray, (M, F)
                    Input matrix of M samples and F features
    y               : jnp.ndarray, (M, 1)
                    Target matrix of M samples and 1 target
    c               : jnp.ndarray, (M, 1)
                    Dual coefficients
    y_mean          : jnp.ndarray, ()
                    Target mean
    kernel          : callable
                    Kernel function
    full_covariance : bool
                    Whether to return also the full posterior covariance
    Returns
    -------
    mu              : jnp.ndarray, (M, 1)
                    Predicted mean
    C_nn            : jnp.ndarray, (M, M)
                    Predicted covariance
    """

    kernel_params, sigma = split_params(params)
    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
    sigma = softplus(sigma)

    K_mn = kernel(x_train, x, kernel_params)
    mu = jnp.dot(c.T, K_mn).reshape(-1, 1) + y_mean

    if full_covariance:
        C_mm = kernel(x_train, x_train, kernel_params) + sigma**2 * jnp.eye(
            x_train.shape[0]
        )
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = kernel(x, x, kernel_params) - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn

    return mu


# Public interface collecting related methods
GaussianProcessRegression = namedtuple(
    "GaussianProcessRegression",
    [
        "lml",
        "fit",
        "predict",
    ],
)(
    _gpr_log_marginal_likelihood,
    _gpr_fit,
    _gpr_predict,
)

# Aliases
GPR = GaussianProcessRegression


# Export
__all__ = [
    'GaussianProcessRegression',
    'GPR',
]
