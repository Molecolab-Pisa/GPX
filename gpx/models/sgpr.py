from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit
from jax._src import prng
from jax.typing import ArrayLike

from ..bijectors import Softplus
from ..kernels.operations import kernel_center, kernel_center_test_test
from ..mean_functions import data_mean, zero_mean
from ..parameters.model_state import ModelState
from ..parameters.parameter import Parameter, is_parameter
from ..priors import NormalPrior
from .base import BaseGP
from .utils import (
    _check_object_is_callable,
    _check_object_is_type,
    _check_recursive_dict_type,
    sample,
)

# =============================================================================
# Sparse Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=[3, 4, 5])
def _log_marginal_likelihood(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
    center_kernel: Optional[bool] = False,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

    H = K_nm (K_mm)⁻¹ K_mn

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value

    m = x_locs.shape[0]
    mu = mean_function(y)
    y = y - mu
    n = y.shape[0]

    K_mm = kernel(x_locs, x_locs, kernel_params)
    K_mn = kernel(x_locs, x, kernel_params)

    if center_kernel:
        k_mean = jnp.mean(K_mm, axis=0)
        K_mm = kernel_center(K_mm, k_mean)
        K_mn = kernel_center(K_mn, k_mean)

    L_m = jsp.linalg.cholesky(K_mm + 1e-10 * jnp.eye(m), lower=True)
    G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
    C_nn = jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.eye(n) + 1e-10 * jnp.eye(n)
    L_n = jsp.linalg.cholesky(C_nn, lower=True)
    cy = jsp.linalg.solve_triangular(L_n, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_n)))
    mll -= n * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / n

    return mll


@partial(jit, static_argnums=[4, 5, 6])
def _log_marginal_likelihood_derivs(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
    center_kernel: Optional[bool] = False,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

    H = K_nm (K_mm)⁻¹ K_mn

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value

    m = x_locs.shape[0]
    mu = mean_function(y)
    y = y - mu
    n = y.shape[0]

    K_mm = kernel.d01kj(x_locs, x_locs, kernel_params, jacobian_locs, jacobian_locs)
    K_mn = kernel.d01kj(x_locs, x, kernel_params, jacobian_locs, jacobian)

    if center_kernel:
        k_mean = jnp.mean(K_mm, axis=0)
        K_mm = kernel_center(K_mm, k_mean)
        K_mn = kernel_center(K_mn, k_mean)

    L_m = jsp.linalg.cholesky(K_mm + 1e-10 * jnp.eye(m), lower=True)
    G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
    C_nn = jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.eye(n) + 1e-10 * jnp.eye(n)
    L_n = jsp.linalg.cholesky(C_nn, lower=True)
    cy = jsp.linalg.solve_triangular(L_n, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_n)))
    mll -= n * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / n

    return mll


def log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

        lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

        H = K_nm (K_mm)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        y: labels
    Returns:
        lml: log marginal likelihood
    """
    return _log_marginal_likelihood(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        mean_function=state.mean_function,
        center_kernel=state.center_kernel,
    )


def log_marginal_likelihood_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

        lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

        H = K_nm (K_mm)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
    Returns:
        lml: log marginal likelihood
    """
    return _log_marginal_likelihood_derivs(
        params=state.params,
        x=x,
        y=y,
        jacobian=jacobian,
        kernel=state.kernel,
        mean_function=zero_mean,
        center_kernel=state.center_kernel,
    )


def log_prior(state: ModelState) -> Array:
    "Computes the log p(θ) assuming independence of θ"
    return jax.tree_util.tree_reduce(
        lambda init, p: init + p.prior.logpdf(p.value),
        state.params,
        initializer=0.0,
        is_leaf=is_parameter,
    )


def log_posterior(state: ModelState, x: ArrayLike, y: ArrayLike) -> Array:
    """Computes the log posterior

        log p(θ|y) = log p(y|θ) + log p(θ)

    where log p(y|θ) is the log marginal likelihood.
    it is assumed that hyperparameters θ are independent.
    """
    return log_marginal_likelihood(state=state, x=x, y=y) + log_prior(state=state)


def log_posterior_derivs(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> Array:
    """Computes the log posterior

        log p(θ|y) = log p(y|θ) + log p(θ)

    where log p(y|θ) is the log marginal likelihood.
    it is assumed that hyperparameters θ are independent.
    """
    return log_marginal_likelihood_derivs(
        state=state, x=x, y=y, jacobian=jacobian
    ) + log_prior(state=state)


def neg_log_marginal_likelihood(state: ModelState, x: ArrayLike, y: ArrayLike) -> Array:
    "Returns the negative log marginal likelihood"
    return -log_marginal_likelihood(state=state, x=x, y=y)


def neg_log_marginal_likelihood_derivs(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> Array:
    "Returns the negative log marginal likelihood"
    return -log_marginal_likelihood_derivs(state=state, x=x, y=y, jacobian=jacobian)


def neg_log_posterior(state: ModelState, x: ArrayLike, y: ArrayLike) -> Array:
    "Returns the negative log posterior"
    return -log_posterior(state=state, x=x, y=y)


def neg_log_posterior_derivs(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> Array:
    "Returns the negative log posterior"
    return -log_posterior_derivs(state=state, x=x, y=y, jacobian=jacobian)


@partial(jit, static_argnums=[3, 4, 5])
def _fit(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
    center_kernel: bool,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    μ = m(y)

    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value

    mu = mean_function(y)
    y = y - mu

    K_mn = kernel(x_locs, x, kernel_params)

    C_mm = kernel(x_locs, x_locs, kernel_params)

    # Here we center over the induced locations
    if center_kernel:
        k_mean = jnp.mean(C_mm, axis=0)
        C_mm = kernel_center(C_mm, k_mean)
        K_mn = kernel_center(K_mn, k_mean)
    else:
        k_mean = None

    C_mm = sigma**2 * C_mm + jnp.dot(K_mn, K_mn.T) + 1e-10 * jnp.eye(x_locs.shape[0])
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y)).reshape(-1, 1)

    return c, mu, k_mean


@partial(jit, static_argnums=[4, 5, 6])
def _fit_derivs(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    jacobian: ArrayLike,
    mean_function: Callable,
    center_kernel: bool,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    μ = 0.

    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value

    mu = mean_function(y)
    y = y - mu

    K_mn = kernel.d01kj(x_locs, x, kernel_params, jacobian_locs, jacobian)

    C_mm = kernel(x_locs, x_locs, kernel_params, jacobian_locs, jacobian_locs)

    # Here we center over the induced locations
    if center_kernel:
        k_mean = jnp.mean(C_mm, axis=0)
        C_mm = kernel_center(C_mm, k_mean)
        K_mn = kernel_center(K_mn, k_mean)
    else:
        k_mean = None

    C_mm = sigma**2 * C_mm + jnp.dot(K_mn, K_mn.T) + 1e-10 * jnp.eye(x_locs.shape[0])
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y)).reshape(-1, 1)

    return c, mu, k_mean


def fit(state: ModelState, x: ArrayLike, y: ArrayLike) -> ModelState:
    """fits a SGPR (projected processes)

        μ = m(y)

        c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    Args:
        state: model state
        x: observations
        y: labels
    Returns:
        state: fitted model state
    """
    c, mu, k_mean = _fit(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        mean_function=state.mean_function,
        center_kernel=state.center_kernel,
    )
    state = state.update(
        dict(x_train=x, y_train=y, c=c, mu=mu, k_mean=k_mean, is_fitted=True)
    )
    return state


def fit_derivs(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> ModelState:
    """fits a SGPR (projected processes)

        μ = 0.

        c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
    Returns:
        state: fitted model state
    """
    c, mu, k_mean = _fit_derivs(
        params=state.params,
        x=x,
        y=y,
        jacobian=jacobian,
        kernel=state.kernel,
        mean_function=zero_mean,
        center_kernel=state.center_kernel,
    )
    state = state.update(
        dict(
            x_train=x,
            y_train=y,
            jacobian_train=jacobian,
            c=c,
            mu=mu,
            k_mean=k_mean,
            is_fitted=True,
        )
    )
    return state


@partial(jit, static_argnums=[4, 5, 6])
def _predict(
    params: Dict[str, Parameter],
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
    center_kernel: Optional[bool] = False,
    k_mean: Optional[ArrayLike] = None,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value

    K_mn = kernel(x_locs, x, kernel_params)

    if center_kernel:
        k_mean_train_test = jnp.mean(K_mn, axis=0)
        K_mn = kernel_center(K_mn, k_mean)

    mu = mu + jnp.dot(c.T, K_mn).reshape(-1, 1)

    if full_covariance:
        m = x_locs.shape[0]
        K_mm = kernel(x_locs, x_locs, kernel_params)
        if center_kernel:
            K_mm = kernel_center(K_mm, k_mean)
        L_m = jsp.linalg.cholesky(K_mm + jnp.eye(m) * 1e-10, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        L_m = jsp.linalg.cholesky(
            (sigma**2 * K_mm + jnp.dot(K_mn, K_mn.T)) + jnp.eye(m) * 1e-10,
            lower=True,
        )
        H_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)

        C_nn = kernel(x, x, kernel_params)

        if center_kernel:
            C_nn = kernel_center_test_test(C_nn, k_mean, k_mean_train_test)

        C_nn = C_nn - jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.dot(H_mn.T, H_mn)
        return mu, C_nn

    return mu


@partial(jit, static_argnums=[4, 5, 6])
def _predict_derivs(
    params: Dict[str, Parameter],
    x: ArrayLike,
    jacobian: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
    center_kernel: Optional[bool] = False,
    k_mean: Optional[ArrayLike] = None,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value

    K_mn = kernel.d01kj(x_locs, x, kernel_params, jacobian_locs, jacobian)

    if center_kernel:
        k_mean_train_test = jnp.mean(K_mn, axis=0)
        K_mn = kernel_center(K_mn, k_mean)

    mu = mu + jnp.dot(c.T, K_mn).reshape(-1, 1)

    if full_covariance:
        m = x_locs.shape[0]
        K_mm = kernel.d01kj(x_locs, x_locs, kernel_params, jacobian_locs, jacobian_locs)
        if center_kernel:
            K_mm = kernel_center(K_mm, k_mean)
        L_m = jsp.linalg.cholesky(K_mm + jnp.eye(m) * 1e-10, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        L_m = jsp.linalg.cholesky(
            (sigma**2 * K_mm + jnp.dot(K_mn, K_mn.T)) + jnp.eye(m) * 1e-10,
            lower=True,
        )
        H_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)

        C_nn = kernel(x, x, kernel_params, jacobian, jacobian)

        if center_kernel:
            C_nn = kernel_center_test_test(C_nn, k_mean, k_mean_train_test)

        C_nn = C_nn - jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.dot(H_mn.T, H_mn)
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
        mu=state.mu,
        kernel=state.kernel,
        full_covariance=full_covariance,
        center_kernel=state.center_kernel,
        k_mean=state.k_mean,
    )


def predict_derivs(
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
    full_covariance: Optional[bool] = False,
) -> Array:
    """predicts with a SGPR (projected processes)

        μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y

        C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        jacobian: jacobian of x
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
        jacobian=jacobian,
        c=state.c,
        mu=0.0,
        kernel=state.kernel,
        full_covariance=full_covariance,
        center_kernel=state.center_kernel,
        k_mean=state.k_mean,
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
    if state.center_kernel:
        k_mean = jnp.mean(cov, axis=0)
        cov = kernel_center(cov, k_mean)
    cov = cov + sigma * jnp.eye(cov.shape[0]) + 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def sample_prior_derivs(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    """samples from the prior of a SGPR (projected processes)

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        jacobian: jacobian of x
        n_samples: number of samples to draw
    Returns:
        samples: samples from the prior distribution
    """
    # not 100% sure that it's the same as the full GP though
    kernel = state.kernel
    kernel_params = state.params["kernel_params"]
    sigma = state.params["sigma"].value

    mean = jnp.zeros(x.shape)
    cov = kernel.d01kj(x, x, kernel_params, jacobian, jacobian)
    if state.center_kernel:
        k_mean = jnp.mean(cov, axis=0)
        cov = kernel_center(cov, k_mean)
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


def sample_posterior_derivs(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    """samples from a posterior of the SGPR (projected processes)

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        jacobian: jacobian of x
        n_samples: number of samples to draw
    Returns:
        samplse: samples from the posterior distribution
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Cannot sample from the posterior if the model is not fitted."
        )
    mean, cov = predict_derivs(state, x=x, jacobian=jacobian, full_covariance=True)
    cov += 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def default_params() -> Dict[str, Parameter]:
    sigma = Parameter(
        value=1.0,
        trainable=True,
        bijector=Softplus(),
        prior=NormalPrior(loc=0.0, scale=1.0),
    )

    return dict(sigma=sigma)


def init(
    kernel: Callable,
    mean_function: Callable = data_mean,
    x_locs: Parameter = None,
    jacobian_locs: Parameter = None,
    kernel_params: Dict[str, Parameter] = None,
    sigma: Parameter = None,
    loss_fn: Callable = neg_log_marginal_likelihood,
    center_kernel: bool = False,
) -> ModelState:
    """initializes the model state of a SGPR (projected processes)

    Args:
        kernel: kernel function
        kernel_params: kernel parameters
        sigma: standard deviation of the gaussian noise
        x_locs: landmark points (support points) of the SGPR
        loss_fn: loss function. Default is negative log marginal likelihood
    Returns:
        state: model state
    """
    _check_object_is_callable(kernel, "kernel")

    if kernel_params is None:
        kernel_params = kernel.default_params()
    else:
        _check_object_is_type(kernel_params, dict, "kernel_params")
        _check_recursive_dict_type(kernel_params, Parameter)

    if sigma is None:
        sigma = default_params().pop("sigma")
    else:
        _check_object_is_type(sigma, Parameter, "sigma")

    _check_object_is_type(x_locs, Parameter, "x_locs")

    params = {"kernel_params": kernel_params, "sigma": sigma, "x_locs": x_locs}
    opt = {
        "loss_fn": loss_fn,
        "is_fitted": False,
        "c": None,
        "mu": None,
        "center_kernel": center_kernel,
        "k_mean": None,
    }

    return ModelState(kernel, mean_function, params, **opt)


# =============================================================================
# Sparse Gaussian Process Regression: interface
# =============================================================================


class SGPR(BaseGP):
    _init_default = dict(is_fitted=False, c=None, y_mean=None, k_mean=None)

    def __init__(
        self,
        kernel: Callable,
        mean_function: Callable = data_mean,
        x_locs: Parameter = None,
        jacobian_locs: Parameter = None,
        kernel_params: Dict[str, Parameter] = None,
        sigma: Parameter = None,
        loss_fn: Callable = neg_log_marginal_likelihood,
        center_kernel: bool = False,
    ) -> None:
        """
        Args:
            kernel: kernel function
            kernel_params: kernel parameters
            sigma: standard deviation of the gaussian noise
            x_locs: landmark points (support points) of SGPR
            jacobian_locs: jacobian of x_locs
            loss_fn: loss function
            center_kernel: whether to center in feature space

        Note:
            It is always required to specify the loss function
            in the definition of the model for training on
            derivatives of the kernel!
        """
        self.state = init(
            kernel=kernel,
            mean_function=mean_function,
            kernel_params=kernel_params,
            sigma=sigma,
            x_locs=x_locs,
            jacobian_locs=jacobian_locs,
            loss_fn=loss_fn,
            center_kernel=center_kernel,
        )

    # parameters
    _default_params_fun = staticmethod(default_params)
    _init_fun = staticmethod(init)
    # lml
    _lml_fun = staticmethod(log_marginal_likelihood)
    _lml_derivs_fun = staticmethod(log_marginal_likelihood_derivs)
    # fit policies
    _fit_fun = staticmethod(fit)
    _fit_derivs_fun = staticmethod(fit_derivs)
    # prediction policies
    _predict_fun = staticmethod(predict)
    _predict_derivs_fun = staticmethod(predict_derivs)
    # sample policies
    _sample_prior_fun = staticmethod(sample_prior)
    _sample_posterior_fun = staticmethod(sample_posterior)
    _sample_prior_derivs_fun = staticmethod(sample_prior_derivs)
    _sample_posterior_derivs_fun = staticmethod(sample_posterior_derivs)
