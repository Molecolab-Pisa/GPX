from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from scipy.optimize import minimize

from ..utils import (
    constrain_parameters,
    unconstrain_parameters,
    split_params,
    print_model,
)


# =============================================================================
# Sparse Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=[4, 5])
def log_marginal_likelihood(params, x, y, x_locs, kernel, return_negative=False):
    """
    Computes the log marginal likelihood for Sparse Gaussian Process Regression
    (projected processes).
    Arguments
    ---------
    params          : dict
                    Dictionary of parameters. Should have a 'kernel_params' keyword
                    to specify kernel parameters (a dictionary) and a 'sigma' keyword
                    to specify the noise.
                    If the input locations should be optimized, they
                    must be included in `params` dictionary under the keyword 'x_locs'.
                    In this case, `x_locs` is ignored.
    x               : jnp.ndarray, (M, F)
                    Input matrix of M samples and F features
    y               : jnp.ndarray, (M, 1)
                    Target matrix of M samples and 1 target
    x_locs          : jnp.ndarray, (N, F)
                    Input locations (N <= M) with F features
    kernel          : callable
                    Kernel function
    return_negative : bool
                    Whether to return the negative marginal log likelihood
    Returns
    -------
    mll             : jnp.ndarray, ()
                    Log marginal likelihood
    """

    kernel_params, sigma = split_params(params)
    x_locs = params["x_locs"] if "x_locs" in params.keys() else x_locs

    y_mean = jnp.mean(y)
    y = y - y_mean
    n = y.shape[0]
    m = x_locs.shape[0]

    K_mm = kernel(x_locs, x_locs, kernel_params)
    K_mn = kernel(x_locs, x, kernel_params)

    L_m = jsp.linalg.cholesky(K_mm + 1e-10 * jnp.eye(m), lower=True)
    G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
    C_nn = jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.eye(n)
    L_n = jsp.linalg.cholesky(C_nn, lower=True)
    cy = jsp.linalg.solve_triangular(L_n, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_n)))
    mll -= n * 0.5 * jnp.log(2.0 * jnp.pi)

    if return_negative:
        return -mll

    return mll


@partial(jit, static_argnums=4)
def fit(params, x, y, x_locs, kernel):
    """
    Fits a Sparse Gaussian Process Regression model (Projected Processes).
    Arguments
    ---------
    params          : dict
                    Dictionary of parameters. Should have a 'kernel_params' keyword
                    to specify kernel parameters (a dictionary) and a 'sigma' keyword
                    to specify the noise. If the input locations should be optimized, they
                    must be included in `params` dictionary under the keyword 'x_locs'.
                    In this case, `x_locs` is ignored.
    x               : jnp.ndarray, (M, F)
                    Input matrix of M samples and F features
    y               : jnp.ndarray, (M, 1)
                    Target matrix of M samples and 1 target
    x_locs          : jnp.ndarray, (N, F)
                    Input locations (N <= M) with F features
    kernel          : callable
                    Kernel function
    Returns
    -------
    c       : jnp.ndarray, (M, 1)
            Dual coefficients
    y_mean  : jnp.ndarray, ()
            Target mean
    """

    kernel_params, sigma = split_params(params)
    x_locs = params["x_locs"] if "x_locs" in params.keys() else x_locs

    y_mean = jnp.mean(y)
    y = y - y_mean

    K_mn = kernel(x_locs, x, kernel_params)
    C_mm = sigma**2 * kernel(x_locs, x_locs, kernel_params) + jnp.dot(K_mn, K_mn.T)
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y)).reshape(-1, 1)

    return c, y_mean


@partial(jit, static_argnums=[5, 6])
def predict(params, x_locs, x, c, y_mean, kernel, full_covariance=False):
    """
    Predicts using a Sparse Gaussian Process Regression model (Projected Processes).
    Arguments
    ---------
    params          : dict
                    Dictionary of parameters. Should have a 'kernel_params' keyword
                    to specify kernel parameters (a dictionary) and a 'sigma' keyword
                    to specify the noise. If the input locations should be optimized, they
                    must be included in `params` dictionary under the keyword 'x_locs'.
                    In this case, `x_locs` is ignored.
    x_locs          : jnp.ndarray, (N, F)
                    Input locations (N <= M) with F features
    x               : jnp.ndarray, (M, F)
                    Input matrix of M samples and F features
    c               : jnp.ndarray, (N, 1)
                    Dual coefficients
    y_mean          : jnp.ndarray, ()
                    Target mean
    kernel          : callable
                    Kernel function
    full_covariance : bool
                    Whether to return the full posterior covariance
    Returns
    -------
    mu              : jnp.ndarray, (M, 1)
                    Posterior mean
    C_nn            : jnp.ndarray, (M, M)
                    Posterior covariance
    """

    kernel_params, sigma = split_params(params)
    x_locs = params["x_locs"] if "x_locs" in params.keys() else x_locs

    K_mn = kernel(x_locs, x, kernel_params)
    mu = jnp.dot(c.T, K_mn).reshape(-1, 1) + y_mean

    if full_covariance:
        m = x_locs.shape[0]
        K_mm = kernel(x_locs, x_locs, kernel_params)
        L_m = jsp.linalg.cholesky(K_mm + jnp.eye(m) * 1e-10, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        L_m = jsp.linalg.cholesky(
            (sigma**2 * K_mm + jnp.dot(K_mn, K_mn.T)) + jnp.eye(m) * 1e-10,
            lower=True,
        )
        H_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = (
            kernel(x, x, kernel_params) - jnp.dot(G_mn.T, G_mn) + jnp.dot(H_mn.T, H_mn)
        )
        return mu, C_nn

    return mu


# =============================================================================
# Sparse Gaussian Process Regression: interface
# =============================================================================


class SparseGaussianProcessRegression:
    def __init__(self, x_locs, kernel, kernel_params, sigma, optimize_locs=False):
        self.kernel = kernel
        self.kernel_params = tree_map(lambda p: jnp.array(p), kernel_params)
        self.sigma = jnp.array(sigma)

        self.params = {"sigma": self.sigma, "kernel_params": self.kernel_params}

        self.optimize_locs = optimize_locs
        if optimize_locs:
            self.params["x_locs"] = jnp.array(x_locs)
            self.x_locs = self.params["x_locs"]

            self.constrain_parameters = partial(constrain_parameters, ignore=["x_locs"])
            self.unconstrain_parameters = partial(
                unconstrain_parameters, ignore=["x_locs"]
            )

        else:
            self.x_locs = jnp.array(x_locs)
            self.constrain_parameters = constrain_parameters
            self.unconstrain_parameters = unconstrain_parameters

        self.params_unconstrained = self.unconstrain_parameters(
            self.params, ignore=["x_locs"]
        )

    def print(self, **kwargs):
        return print_model(self, **kwargs)

    def log_marginal_likelihood(self, x, y, return_negative=False):
        return log_marginal_likelihood(
            self.params,
            x=x,
            y=y,
            x_locs=self.x_locs,
            kernel=self.kernel,
            return_negative=return_negative,
        )

    def fit(self, x, y):
        x0, unravel_fn = ravel_pytree(self.params_unconstrained)

        def loss(xt):
            params = unravel_fn(xt)
            params = self.constrain_parameters(params)
            return log_marginal_likelihood(
                params,
                x=x,
                y=y,
                x_locs=self.x_locs,
                kernel=self.kernel,
                return_negative=True,
            )

        grad_loss = grad(loss)

        optres = minimize(loss, x0, method="L-BFGS-B", jac=grad_loss)

        self.params_unconstrained = unravel_fn(optres.x)
        self.params = self.constrain_parameters(self.params_unconstrained)

        self.optimize_results_ = optres

        self.c_, self.y_mean_ = fit(
            self.params,
            x=x,
            y=y,
            x_locs=self.x_locs,
            kernel=self.kernel,
        )

        return self

    def predict(self, x, full_covariance=False):
        # TODO: add prediction using prior only
        return predict(
            self.params,
            x_locs=self.x_locs,
            x=x,
            c=self.c_,
            y_mean=self.y_mean_,
            kernel=self.kernel,
            full_covariance=full_covariance,
        )


# Alias
SGPR = SparseGaussianProcessRegression


# Export
__all__ = [
    "SparseGaussianProcessRegression",
    "SGPR",
]


# def sgpr_optimize(
#   params,
#   x,
#   y,
#   x_locs,
#   kernel,
#   n_steps=100,
#   step_size=0.01,
#   verbose=20,
# ):
#   '''
#   Optimize a Sparse Gaussian Process Regression model (Projected Processes).
#   Arguments
#   ---------
#   params          : dict
#                   Dictionary of parameters. Should have a 'kernel_params' keyword
#                   to specify kernel parameters (a dictionary) and a 'sigma' keyword
#                   to specify the noise. If the input locations should be optimized, they
#                   must be included in `params` dictionary under the keyword 'x_locs'.
#                   In this case, `x_locs` is ignored.
#   x               : jnp.ndarray, (M, F)
#                   Input matrix of M samples and F features
#   y               : jnp.ndarray, (N, 1)
#                   Target matrix with M samples and 1 target
#   x_locs          : jnp.ndarray, (N, F)
#                   Input locations (N <= M) with F features
#   kernel          : callable
#                   Kernel function
#   n_steps         : int
#                   Number of optimization steps
#   step_size       : float
#                   Step size / learning rate
#   verbose         : int
#                   Frequency for printing the loss (negative log marginal likelihood)
#   Returns
#   -------
#   params          : dict
#                   Optimized parameters
#   '''
#
#   opt_init, opt_update, get_params = jax_optim.adam(step_size=step_size)
#   opt_state = opt_init(params)
#   loss_fn = partial(sgpr_log_marginal_likelihood, kernel=kernel, return_negative=True)
#
#   @jit
#   def train_step(step_i, opt_state, x, y, x_locs):
#       params = get_params(opt_state)
#       grads = grad(loss_fn, argnums=0)(params, x, y, x_locs)
#       return opt_update(step_i, grads, opt_state)
#
#   for step_i in range(n_steps):
#       opt_state = train_step(step_i, opt_state, x, y, x_locs)
#       if step_i % verbose == 0:
#           params = get_params(opt_state)
#           loss = loss_fn(params, x, y, x_locs)
#           print(" loss : {:.3f}".format(float(loss)))
#
#   params = get_params(opt_state)
#
#   return params
