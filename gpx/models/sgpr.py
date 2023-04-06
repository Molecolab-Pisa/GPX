from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit
from jax._src import prng
from jax.typing import ArrayLike
from typing_extensions import Self

from ..optimize import scipy_minimize
from ..parameters.model_state import ModelState
from ..parameters.parameter import Parameter, parse_param
from .utils import sample

# =============================================================================
# Sparse Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=[3, 4])
def _log_marginal_likelihood(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    return_negative: Optional[bool] = False,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

    H = K_nm (K_mm)⁻¹ K_mn

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value

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


def log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    return_negative: Optional[bool] = False,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

        lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

        H = K_nm (K_mm)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        y: labels
        return_negative: whether to return the negative of the lml
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
    params: Dict[str, Parameter], x: ArrayLike, y: ArrayLike, kernel: Callable
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    y_mean = (1/n) Σ_i y_i

    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value

    y_mean = jnp.mean(y)
    y = y - y_mean

    K_mn = kernel(x_locs, x, kernel_params)
    C_mm = (
        sigma**2 * kernel(x_locs, x_locs, kernel_params)
        + jnp.dot(K_mn, K_mn.T)
        + 1e-10 * jnp.eye(x_locs.shape[0])
    )
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y)).reshape(-1, 1)

    return c, y_mean


def fit(state: ModelState, x: ArrayLike, y: ArrayLike) -> ModelState:
    """fits a SGPR (projected processes)

        y_mean = (1/n) Σ_i y_i

        c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

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


@partial(jit, static_argnums=[4, 5])
def _predict(
    params: Dict[str, Parameter],
    x: ArrayLike,
    c: ArrayLike,
    y_mean: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value

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
            kernel(x, x, kernel_params)
            - jnp.dot(G_mn.T, G_mn)
            + sigma**2 * jnp.dot(H_mn.T, H_mn)
        )
        return mu, C_nn

    return mu


def predict(
    state: ModelState, x: ArrayLike, full_covariance: Optional[bool] = False
) -> Array:
    """predicts with a SGPR (projected processes)

        μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

        C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        full_covariance: whether to return the covariance matrix too
    Returns:
        μ: predicted mean
        C_nn (optional): predicted covariance
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Model is not fitted. Run 'fit' to fit the model before prediction."
        )
    return _predict(
        params=state.params,
        x=x,
        c=state.c,
        y_mean=state.y_mean,
        kernel=state.kernel,
        full_covariance=full_covariance,
    )


def sample_prior(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    """samples from the prior of a SGPR (projected processes)

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        n_samples: number of samples to draw
    Returns:
        samples: samples from the prior distribution
    """
    # not 100% sure that it's the same as the full GP though
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
    x: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    """samples from a posterior of the SGPR (projected processes)

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        n_samples: number of samples to draw
    Returns:
        samplse: samples from the posterior distribution
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Cannot sample from the posterior if the model is not fitted."
        )
    mean, cov = predict(state, x=x, full_covariance=True)
    cov += 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def init(
    kernel: Callable, kernel_params: Dict[str, Tuple], sigma: Tuple, x_locs: Tuple
) -> ModelState:
    """initializes the model state of a SGPR (projected processes)

    Args:
        kernel: kernel function
        kernel_params: kernel parameters
        sigma: standard deviation of the gaussian noise
        x_locs: landmark points (support points) of the SGPR
    Returns:
        state: model state
    """
    if not callable(kernel):
        raise RuntimeError(
            f"kernel must be provided as a callable function, you provided"
            f" {type(kernel)}"
        )

    if not isinstance(kernel_params, dict):
        raise RuntimeError(
            f"kernel_params must be provided as a dictionary, you provided"
            f" {type(kernel_params)}"
        )

    kp = {}
    for key in kernel_params:
        param = kernel_params[key]
        kp[key] = parse_param(param)

    sigma = parse_param(sigma)
    x_locs = parse_param(x_locs)
    params = {"kernel_params": kp, "sigma": sigma, "x_locs": x_locs}
    opt = dict(is_fitted=False, c=None, y_mean=None)

    return ModelState(kernel, params, **opt)


# =============================================================================
# Sparse Gaussian Process Regression: interface
# =============================================================================


class SparseGaussianProcessRegression:
    def __init__(
        self,
        kernel: Callable,
        kernel_params: Dict[str, Tuple],
        sigma: Tuple,
        x_locs: Tuple,
    ) -> None:
        """
        Args:
            kernel: kernel function
            kernel_params: kernel parameters
            sigma: standard deviation of the gaussian noise
            x_locs: landmark points (support points) of SGPR
        """
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.sigma = sigma
        self.x_locs = x_locs
        self.state = init(
            kernel=kernel, kernel_params=kernel_params, sigma=sigma, x_locs=x_locs
        )

    def print(self) -> None:
        "prints the model parameters"
        return self.state.print_params()

    def log_marginal_likelihood(
        self, x: ArrayLike, y: ArrayLike, return_negative: Optional[bool] = False
    ) -> Array:
        """log marginal likelihood for SGPR (projected processes)

            lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

            H = K_nm (K_mm)⁻¹ K_mn

        Args:
            x: observations
            y: labels
            return_negative: whether to return the negative of the lml
        """
        return log_marginal_likelihood(
            self.state,
            x=x,
            y=y,
            return_negative=return_negative,
        )

    def fit(
        self, x: ArrayLike, y: ArrayLike, minimize_lml: Optional[bool] = True
    ) -> Self:
        """fits a SGPR (projected processes)

            y_mean = (1/n) Σ_i y_i

            c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

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
        self.x_locs_ = self.state.params["x_locs"].value

        return self

    def predict(self, x: ArrayLike, full_covariance: Optional[bool] = False) -> Array:
        """predicts with a SGPR (projected processes)

            μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

            C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

        Args:
            x: observations
            full_covariance: whether to return the covariance matrix too
        """
        if not hasattr(self, "c_"):
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} is not fitted yet."
                "Call 'fit' before using this model for prediction."
            )
        return predict(self.state, x=x, full_covariance=full_covariance)

    def sample(
        self,
        key: prng.PRNGKeyArray,
        x: ArrayLike,
        n_samples: Optional[int] = 1,
        kind: Optional[str] = "prior",
    ) -> Array:
        """draws samples from a SGPR (projected processes)

        Args:
            key: JAX PRNGKey
            x: observations
            n_samples: number of samples to draw
            kind: whether to draw samples from the prior ('prior')
                  or from the posterior ('posterior')
        """
        if kind == "prior":
            return sample_prior(key, state=self.state, x=x, n_samples=n_samples)
        elif kind == "posterior":
            return sample_posterior(key, state=self.state, x=x, n_samples=n_samples)
        else:
            raise ValueError(
                f"kind can be either 'prior' or 'posterior', you provided {kind}"
            )


# Alias
SGPR = SparseGaussianProcessRegression
