from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit  # , grad

# from jax.tree_util import tree_map
# from jax.flatten_util import ravel_pytree

# from scipy.optimize import minimize

from .utils import sample
from ..parameters import ModelState
from ..parameters.parameter import parse_param


# =============================================================================
# Standard Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=[3, 4])
def _log_marginal_likelihood(params, x, y, kernel, return_negative=False):
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

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


def log_marginal_likelihood(state, x, y, return_negative=False):
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
    return _log_marginal_likelihood(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        return_negative=return_negative,
    )


@partial(jit, static_argnums=[3])
def _fit(params, x, y, kernel):
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    y_mean = jnp.mean(y)
    y = y - y_mean

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(y.shape[0])
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)

    return c, y_mean


def fit(state, x, y):
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
    c, y_mean = _fit(params=state.params, x=x, y=y, kernel=state.kernel)
    state = state.update(dict(c=c, y_mean=y_mean, is_fitted=True))
    return state


@partial(jit, static_argnums=[5, 6])
def _predict(
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

    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

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


def predict(state, x_train, x, full_covariance=False):
    if not state.is_fitted:
        raise RuntimeError(
            "Model is not fitted. Run `fit` to fit the model before prediction."
        )
    return _predict(
        params=state.params,
        x_train=x_train,
        x=x,
        c=state.c,
        y_mean=state.y_mean,
        kernel=state.kernel,
        full_covariance=full_covariance,
    )


def sample_prior(key, state, x, n_samples=1):
    kernel = state.kernel
    kernel_params = state.params["kernel_params"]
    sigma = state.params["sigma"].value

    mean = jnp.zeros(x.shape)
    cov = kernel(x, x, kernel_params)
    cov = cov + sigma * jnp.eye(cov.shape[0]) + 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def sample_posterior(key, state, x_train, x, n_samples=1):
    if not state.is_fitted:
        raise RuntimeError(
            "Cannot sample from the posterior if the model is not fitted"
        )

    mean, cov = predict(state, x_train, x, full_covariance=True)
    cov += 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def init(kernel, kernel_params, sigma):

    # validate kernel argument
    if not callable(kernel):
        raise RuntimeError(
            f"kernel must be provided as a callable function, you provided {type(kernel)}"
        )

    # validate kernel_params argument
    if not isinstance(kernel_params, dict):
        raise RuntimeError(
            f"kernel_params must be provided as a dictionary, you provided {type(kernel_params)}"
        )

    kp = {}
    for key in kernel_params:
        param = kernel_params[key]
        kp[key] = parse_param(param)

    # validate sigma argument
    sigma = parse_param(sigma)

    # merge kernel parameters and sigma
    params = {"kernel_params": kp, "sigma": sigma}

    # additional fields
    opt = dict(is_fitted=False, c=None, y_mean=None)

    return ModelState(kernel, params, **opt)


# #=============================================================================
# # Standard Gaussian Process Regression: interface
# # =============================================================================
#
#
# class GaussianProcessRegression:
#     def __init__(self, kernel, kernel_params, sigma):
#         self.kernel = kernel
#         self.kernel_params = tree_map(lambda p: jnp.array(p), kernel_params)
#         self.sigma = jnp.array(sigma)
#
#         self.constrain_parameters = constrain_parameters
#         self.unconstrain_parameters = unconstrain_parameters
#
#         self.params = {"sigma": self.sigma, "kernel_params": self.kernel_params}
#         self.params_unconstrained = self.unconstrain_parameters(self.params)
#
#     def print(self, **kwargs):
#         return print_model(self, **kwargs)
#
#     def log_marginal_likelihood(self, x, y, return_negative=False):
#         return log_marginal_likelihood(
#             self.params, x=x, y=y, kernel=self.kernel, return_negative=return_negative
#         )
#
#     def fit(self, x, y):
#         x0, unravel_fn = ravel_pytree(self.params_unconstrained)
#
#         def loss(xt):
#             params = unravel_fn(xt)
#             params = self.constrain_parameters(params)
#             return log_marginal_likelihood(
#                 params, x=x, y=y, kernel=self.kernel, return_negative=True
#             )
#
#         grad_loss = jit(grad(loss))
#
#         optres = minimize(loss, x0, method="L-BFGS-B", jac=grad_loss)
#
#         self.params_unconstrained = unravel_fn(optres.x)
#         self.params = self.constrain_parameters(self.params_unconstrained)
#
#         self.optimize_results_ = optres
#
#         self.c_, self.y_mean_ = fit(self.params, x=x, y=y, kernel=self.kernel)
#         self.x_train = x
#
#         return self
#
#     def predict(self, x, full_covariance=False):
#         if not hasattr(self, "c_"):
#             # not trained, return prior values
#             y_mean = jnp.zeros(x.shape)
#             if full_covariance:
#                 cov = self.kernel(x, x, self.params["kernel_params"])
#                 cov = cov + self.params["sigma"] * jnp.eye(cov.shape[0])
#                 return y_mean, cov
#             return y_mean
#
#         return predict(
#             self.params,
#             x_train=self.x_train,
#             x=x,
#             c=self.c_,
#             y_mean=self.y_mean_,
#             kernel=self.kernel,
#             full_covariance=full_covariance,
#         )
#
#     def sample(self, key, x, n_samples=1):
#         return sample(key, self, x, n_samples=n_samples)
#
#
# # Alias
# GPR = GaussianProcessRegression
#
#
# # Export
# __all__ = [
#     "GaussianProcessRegression",
#     "GPR",
# ]
