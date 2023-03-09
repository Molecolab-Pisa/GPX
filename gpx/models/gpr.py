from __future__ import annotations

from typing import Callable, Tuple, Dict, Optional
from typing_extensions import Self
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from jax._src import prng

from .utils import sample
from ..parameters import ModelState
from ..parameters.parameter import parse_param, Parameter
from ..optimize import scipy_minimize


# =============================================================================
# Standard Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=[3, 4])
def _log_marginal_likelihood(
    params: Dict[str, Parameter],
    x: jnp.ndarray,
    y: jnp.ndarray,
    kernel: Callable,
    return_negative: Optional[bool] = False,
) -> jnp.ndarray:
    """log marginal likelihood for standard gaussian process

    lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)

    """
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
    state: ModelState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    return_negative: Optional[bool] = False,
) -> jnp.ndarray:
    """computes the log marginal likelihood for standard gaussian process

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)

    Args:
        state: model state
        x: observations
        y: labels
        return_negative: whether to return the negative value of the lml
    Returns:
        lml: log marginal likelihood
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
    params: Dict[str, Parameter], x: jnp.ndarray, y: jnp.ndarray, kernel: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """fits a standard gaussian process

    y_mean = (1/n) Σ_i y_i

    c = (K_nn + σ²I)⁻¹y

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    y_mean = jnp.mean(y)
    y = y - y_mean

    C_mm = (
        kernel(x, x, kernel_params)
        + sigma**2 * jnp.eye(y.shape[0])
        + 1e-10 * jnp.eye(y.shape[0])
    )
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)

    return c, y_mean


def fit(state: ModelState, x: jnp.ndarray, y: jnp.ndarray) -> ModelState:
    """fits a standard gaussian process

        y_mean = (1/n) Σ_i y_i

        c = (K_nn + σ²I)⁻¹y

    Args:
        state: model state
        x: observations
        y: labels
    Returns:
        state: fitted model state
    """
    c, y_mean = _fit(params=state.params, x=x, y=y, kernel=state.kernel)
    state = state.update(dict(c=c, y_mean=y_mean, is_fitted=True))
    return state


@partial(jit, static_argnums=[5, 6])
def _predict(
    params: Dict[str, Parameter],
    x_train: jnp.ndarray,
    x: jnp.ndarray,
    c: jnp.ndarray,
    y_mean: jnp.ndarray,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
) -> jnp.ndarray:
    """predicts with standard gaussian process

    μ = K_nm (K_mm + σ²)⁻¹y

    C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

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


def predict(
    state: ModelState,
    x_train: jnp.ndarray,
    x: jnp.ndarray,
    full_covariance: Optional[bool] = False,
) -> jnp.ndarray:
    """predicts with standard gaussian process

        μ = K_nm (K_mm + σ²)⁻¹y

        C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

    Args:
        state: model state
        x_train: train observations
        x: observations
        full_covariance: whether to return the covariance matrix too
    Returns:
        μ: predicted mean
        C_nn: predicted covariance
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
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: jnp.ndarray,
    n_samples: Optional[int] = 1,
) -> jnp.ndarray:
    """returns samples from the prior of a gaussian process

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        n_samples: number of samples to draw

    Returns:
        samples: samples from the prior distribution
    """
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
    x_train: jnp.ndarray,
    x: jnp.ndarray,
    n_samples: Optional[int] = 1,
) -> jnp.ndarray:
    """returns samples from the posterior of a gaussian process

    Args:
        key: JAX PRNGKey
        state: model state
        x_train: train observations
        x: observations
        n_samples: number of samples to draw

    Returns:
        samples: samples from the posterior distribution
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Cannot sample from the posterior if the model is not fitted"
        )
    mean, cov = predict(state, x_train=x_train, x=x, full_covariance=True)
    cov += 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def init(kernel: Callable, kernel_params: Dict[str, Tuple], sigma: Tuple) -> ModelState:
    """initializes the model state of a gaussian process

    Args:
        kernel: kernel function
        kernel_params: kernel parameters
        sigma: standard deviation of gaussian noise
    Returns:
        state: model state
    """
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


# =============================================================================
# Standard Gaussian Process Regression: interface
# =============================================================================


class GaussianProcessRegression:
    def __init__(
        self, kernel: Callable, kernel_params: Dict[str, Tuple], sigma: Tuple
    ) -> None:
        """
        Args:
            kernel: kernel function
            kernel_params: kernel parameters
            sigma: standard deviation of the gaussian noise
        """
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.sigma = sigma
        self.state = init(kernel=kernel, kernel_params=kernel_params, sigma=sigma)

    def print(self) -> None:
        "prints the model parameters"
        return self.state.print_params()

    def log_marginal_likelihood(
        self, x: jnp.ndarray, y: jnp.ndarray, return_negative: Optional[bool] = False
    ) -> jnp.ndarray:
        """log marginal likelihood for standard gaussian process

            lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)

        Args:
            x: observations
            y: labels
            return_negative: whether to return the negative of the lml
        """
        return log_marginal_likelihood(
            self.state, x=x, y=y, return_negative=return_negative
        )

    def fit(
        self, x: jnp.ndarray, y: jnp.ndarray, minimize_lml: Optional[bool] = True
    ) -> Self:
        """fits a standard gaussian process

            y_mean = (1/n) Σ_i y_i

            c = (K_nn + σ²I)⁻¹y

        Args:
            x: observations
            y: labels
            minimize_lml: whether to tune the parameters to optimize the
                          log marginal likelihood
        """
        if minimize_lml:
            loss_fn = partial(log_marginal_likelihood, return_negative=True)
            self.state, optres = scipy_minimize(self.state, x=x, y=y, loss_fn=loss_fn)
            self.optimize_results_ = optres

        self.state = fit(self.state, x=x, y=y)

        self.c_ = self.state.c
        self.y_mean_ = self.state.y_mean
        self.x_train = x

        return self

    def predict(
        self, x: jnp.ndarray, full_covariance: Optional[bool] = False
    ) -> jnp.ndarray:
        """predicts with standard gaussian process

            μ = K_nm (K_mm + σ²)⁻¹y

            C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

        Args:
            x: observations
            full_covariance: whether to return the covariance matrix too
        Returns:
            μ: predicted mean
            C_nn: predicted covariance
        """
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
        x: jnp.ndarray,
        n_samples: Optional[int] = 1,
        kind: Optional[str] = "prior",
    ) -> jnp.ndarray:
        """draws samples from a gaussian process

        Args:
            key: JAX PRNGKey
            x: observations
            n_samples: number of samples to draw
            kind: whether to draw samples from the prior ('prior')
                  or from the posterior ('posterior')
        Returns:
            samples: drawn samples
        """
        if kind == "prior":
            return sample_prior(key, state=self.state, x=x, n_samples=n_samples)
        elif kind == "posterior":
            return sample_posterior(
                key, state=self.state, x_train=self.x_train, x=x, n_samples=n_samples
            )
        else:
            raise ValueError(
                f"kind can be either 'prior' or 'posterior', you provided {kind}"
            )


# Alias
GPR = GaussianProcessRegression
