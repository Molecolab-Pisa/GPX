from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from scipy.optimize import minimize

from .utils import sample
from ..utils import (
    constrain_parameters,
    unconstrain_parameters,
    split_params,
    print_model,
)


# =============================================================================
# Standard Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=3)
def log_marginal_likelihood(params, x, y, kernel, return_negative=False):
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

    y_mean = jnp.mean(y)
    y = y - y_mean
    m = y.shape[0]

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(m) + 1e-10 * jnp.eye(m)
    L_m = jsp.linalg.cholesky(C_mm, lower=True)
    cy = jsp.linalg.solve_triangular(L_m, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_m)))
    mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)

    if return_negative:
        return -mll

    return mll


@partial(jit, static_argnums=3)
def fit(params, x, y, kernel):
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

    y_mean = jnp.mean(y)
    y = y - y_mean

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(y.shape[0])
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)

    return c, y_mean


@partial(jit, static_argnums=5)
def predict(
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

    K_mn = kernel(x_train, x, kernel_params)
    mu = jnp.dot(c.T, K_mn).reshape(-1, 1) + y_mean

    if full_covariance:
        C_mm = kernel(x_train, x_train, kernel_params) + sigma**2 * jnp.eye(
            K_mn.shape[0]
        )
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = kernel(x, x, kernel_params) - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn

    return mu


# =============================================================================
# Standard Gaussian Process Regression: interface
# =============================================================================


class GaussianProcessRegression:
    def __init__(self, kernel, kernel_params, sigma):
        self.kernel = kernel
        self.kernel_params = tree_map(lambda p: jnp.array(p), kernel_params)
        self.sigma = jnp.array(sigma)

        self.constrain_parameters = constrain_parameters
        self.unconstrain_parameters = unconstrain_parameters

        self.params = {"sigma": self.sigma, "kernel_params": self.kernel_params}
        self.params_unconstrained = self.unconstrain_parameters(self.params)

    def print(self, **kwargs):
        return print_model(self, **kwargs)

    def log_marginal_likelihood(self, x, y, return_negative=False):
        return log_marginal_likelihood(
            self.params, x=x, y=y, kernel=self.kernel, return_negative=return_negative
        )

    def fit(self, x, y):
        x0, unravel_fn = ravel_pytree(self.params_unconstrained)

        def loss(xt):
            params = unravel_fn(xt)
            params = self.constrain_parameters(params)
            return log_marginal_likelihood(
                params, x=x, y=y, kernel=self.kernel, return_negative=True
            )

        grad_loss = jit(grad(loss))

        optres = minimize(loss, x0, method="L-BFGS-B", jac=grad_loss)

        self.params_unconstrained = unravel_fn(optres.x)
        self.params = self.constrain_parameters(self.params_unconstrained)

        self.optimize_results_ = optres

        self.c_, self.y_mean_ = fit(self.params, x=x, y=y, kernel=self.kernel)
        self.x_train = x

        return self

    def predict(self, x, full_covariance=False):
        if not hasattr(self, "c_"):
            # not trained, return prior values
            y_mean = jnp.zeros(x.shape)
            if full_covariance:
                cov = self.kernel(x, x, self.params["kernel_params"])
                cov = cov + self.params["sigma"] * jnp.eye(cov.shape[0])
                return y_mean, cov
            return y_mean

        return predict(
            self.params,
            x_train=self.x_train,
            x=x,
            c=self.c_,
            y_mean=self.y_mean_,
            kernel=self.kernel,
            full_covariance=full_covariance,
        )

    def sample(self, key, x, n_samples=1):
        return sample(key, self, x, n_samples=n_samples)


# Alias
GPR = GaussianProcessRegression


# Export
__all__ = [
    "GaussianProcessRegression",
    "GPR",
]
