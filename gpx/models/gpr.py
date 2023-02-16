# from __future__ import annotations

from typing import Any, Callable, Tuple, Dict, Optional
from typing_extensions import Self
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit
from jax._src import prng

from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from .utils import sample
from ..parameters import ModelState
from ..parameters.parameter import parse_param, Parameter

Array = Any


# =============================================================================
# Standard Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=[3, 4])
def _log_marginal_likelihood(
    params: Dict[str, Parameter],
    x: Array,
    y: Array,
    kernel: Callable,
    return_negative: Optional[bool] = False,
) -> Array:
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


def log_marginal_likelihood(
    state: ModelState, x: Array, y: Array, return_negative: Optional[bool] = False
) -> Array:
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
def _fit(
    params: Dict[str, Parameter], x: Array, y: Array, kernel: Callable
) -> Tuple[Array, Array]:
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    y_mean = jnp.mean(y)
    y = y - y_mean

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(y.shape[0])
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)

    return c, y_mean


def fit(state: ModelState, x: Array, y: Array) -> ModelState:
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
    params: Dict[str, Parameter],
    x_train: Array,
    x: Array,
    c: Array,
    y_mean: Array,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
) -> Array:
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


def predict(
    state: ModelState, x_train: Array, x: Array, full_covariance: Optional[bool] = False
) -> Array:
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


def sample_prior(
    key: prng.PRNGKeyArray, state: ModelState, x: Array, n_samples: Optional[int] = 1
) -> Array:
    kernel = state.kernel
    kernel_params = state.params["kernel_params"]
    sigma = state.params["sigma"].value

    mean = jnp.zeros(x.shape)
    cov = kernel(x, x, kernel_params)
    cov = cov + sigma * jnp.eye(cov.shape[0]) + 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def sample_posterior(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x_train: Array,
    x: Array,
    n_samples: Optional[int] = 1,
) -> Array:
    if not state.is_fitted:
        raise RuntimeError(
            "Cannot sample from the posterior if the model is not fitted"
        )
    mean, cov = predict(state, x_train=x_train, x=x, full_covariance=True)
    cov += 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def init(kernel: Callable, kernel_params: Dict[str, Tuple], sigma: Tuple) -> ModelState:
    if not callable(kernel):
        raise RuntimeError(
            f"kernel must be provided as a callable function, you provided {type(kernel)}"
        )

    if not isinstance(kernel_params, dict):
        raise RuntimeError(
            f"kernel_params must be provided as a dictionary, you provided {type(kernel_params)}"
        )

    kp = {}
    for key in kernel_params:
        param = kernel_params[key]
        kp[key] = parse_param(param)

    sigma = parse_param(sigma)
    params = {"kernel_params": kp, "sigma": sigma}
    opt = dict(is_fitted=False, c=None, y_mean=None)

    return ModelState(kernel, params, **opt)


def optimize(
    state: ModelState, x: Array, y: Array
) -> Tuple[ModelState, OptimizeResult]:
    def forward(xt):
        return jnp.array(
            [fwd(x) for fwd, x in zip(state.params_forward_transforms, xt)]
        )

    def backward(xt):
        return jnp.array(
            [bwd(x) for bwd, x in zip(state.params_backward_transforms, xt)]
        )

    x0, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    x0 = backward(x0)

    def loss(xt, state):
        # important: here we first reconstruct the model state with the
        # updated parameters before feeding it to the loss (lml).
        # this ensures that gradients are stopped for parameter with
        # trainable = False.
        xt = forward(xt)
        params = unravel_fn(xt)
        state = state.update(dict(params=params))
        return log_marginal_likelihood(state, x, y, return_negative=True)

    grad_loss = jit(grad(loss))
    optres = minimize(loss, x0=x0, args=(state), method="L-BFGS-B", jac=grad_loss)

    xf = forward(optres.x)
    params = unravel_fn(xf)

    state = state.update(dict(params=params))
    state = fit(state, x=x, y=y)

    return state, optres


# =============================================================================
# Standard Gaussian Process Regression: interface
# =============================================================================


class GaussianProcessRegression:
    def __init__(
        self, kernel: Callable, kernel_params: Dict[str, Tuple], sigma: Tuple
    ) -> None:
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.sigma = sigma
        self.state = init(kernel=kernel, kernel_params=kernel_params, sigma=sigma)

    def print(self) -> None:
        return self.state.print_params()

    def log_marginal_likelihood(
        self, x: Array, y: Array, return_negative: Optional[bool] = False
    ) -> Array:
        return log_marginal_likelihood(
            self.state, x=x, y=y, return_negative=return_negative
        )

    def fit(self, x: Array, y: Array, minimize_lml: Optional[bool] = True) -> Self:
        if minimize_lml:
            self.state, optres = optimize(self.state, x=x, y=y)
            self.optimize_results_ = optres
        else:
            self.state = fit(self.state, x=x, y=y)

        self.c_ = self.state.c
        self.y_mean_ = self.state.y_mean
        self.x_train = x

        return self

    def predict(self, x: Array, full_covariance: Optional[bool] = False) -> Array:
        if not hasattr(self, "c_"):
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} is not fitted yet."
                "Call 'fit' before using this model for prediction."
            )
        return predict(self.state, self.x_train, x=x, full_covariance=full_covariance)

    def sample(
        self,
        key: prng.PRNGKeyArray,
        x: Array,
        n_samples: Optional[int] = 1,
        kind: Optional[str] = "prior",
    ) -> Array:
        if kind == "prior":
            return sample_prior(key, state=self.state, x=x, n_samples=n_samples)
        elif kind == "posterior":
            return sample_posterior(key, state=self.state, x=x, n_samples=n_samples)
        else:
            raise ValueError(
                f"kind can be either 'prior' or 'posterior', you provided {kind}"
            )


# Alias
GPR = GaussianProcessRegression
