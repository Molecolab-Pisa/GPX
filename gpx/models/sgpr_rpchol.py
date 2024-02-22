from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..bijectors import Softplus
from ..defaults import gpxargs
from ..mean_functions import data_mean, zero_mean
from ..parameters.model_state import ModelState
from ..parameters.parameter import Parameter, is_parameter
from ..priors import NormalPrior
from ._sgpr_rpchol import (
    _fit_dense,
    _fit_derivs_dense,
    _fit_derivs_iter,
    _fit_iter,
    _lml_dense,
    _lml_derivs_dense,
    _lml_derivs_iter,
    _lml_iter,
    _predict_dense,
    _predict_derivs_dense,
    _predict_derivs_iter,
    _predict_iter,
    _predict_y_derivs_dense,
    _predict_y_derivs_iter,
)
from .base import BaseGP
from .utils import (
    _check_object_is_callable,
    _check_object_is_type,
    _check_recursive_dict_type,
    sample,
)

KeyArray = Array

# =============================================================================
# Sparse Gaussian Process Regression: functions
# =============================================================================


def log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
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
    if iterative:
        return _lml_iter(
            params=state.params,
            x=x,
            y=y,
            key=state.key,
            n_locs=int(state.n_locs),
            kernel=state.kernel,
            mean_function=state.mean_function,
            num_evals=num_evals,
            num_lanczos=num_lanczos,
            lanczos_key=lanczos_key,
        )
    return _lml_dense(
        params=state.params,
        x=x,
        y=y,
        key=state.key,
        n_locs=int(state.n_locs),
        kernel=state.kernel,
        mean_function=state.mean_function,
    )


def log_marginal_likelihood_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
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
    if iterative:
        return _lml_derivs_iter(
            params=state.params,
            x=x,
            y=y,
            key=state.key,
            n_locs=int(state.n_locs),
            jacobian=jacobian,
            kernel=state.kernel,
            mean_function=zero_mean,
            num_evals=num_evals,
            num_lanczos=num_lanczos,
            lanczos_key=lanczos_key,
        )
    return _lml_derivs_dense(
        params=state.params,
        x=x,
        y=y,
        key=state.key,
        n_locs=int(state.n_locs),
        jacobian=jacobian,
        kernel=state.kernel,
        mean_function=zero_mean,
    )


def log_prior(state: ModelState) -> Array:
    "Computes the log p(θ) assuming independence of θ"
    return jax.tree_util.tree_reduce(
        lambda init, p: init + p.prior.logpdf(p.value),
        state.params,
        initializer=0.0,
        is_leaf=is_parameter,
    )


def log_posterior(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
) -> Array:
    """Computes the log posterior

        log p(θ|y) = log p(y|θ) + log p(θ)

    where log p(y|θ) is the log marginal likelihood.
    it is assumed that hyperparameters θ are independent.
    """
    return log_marginal_likelihood(
        state=state,
        x=x,
        y=y,
        iterative=iterative,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
    ) + log_prior(state=state)


def log_posterior_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
) -> Array:
    """Computes the log posterior

        log p(θ|y) = log p(y|θ) + log p(θ)

    where log p(y|θ) is the log marginal likelihood.
    it is assumed that hyperparameters θ are independent.
    """
    return log_marginal_likelihood_derivs(
        state=state,
        x=x,
        y=y,
        jacobian=jacobian,
        iterative=iterative,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        lanczos_key=lanczos_key,
    ) + log_prior(state=state)


def neg_log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
) -> Array:
    "Returns the negative log marginal likelihood"
    return -log_marginal_likelihood(
        state=state,
        x=x,
        y=y,
        iterative=iterative,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        lanczos_key=lanczos_key,
    )


def neg_log_marginal_likelihood_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
) -> Array:
    "Returns the negative log marginal likelihood"
    return -log_marginal_likelihood_derivs(
        state=state,
        x=x,
        y=y,
        jacobian=jacobian,
        iterative=iterative,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        lanczos_key=lanczos_key,
    )


def neg_log_posterior(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
) -> Array:
    "Returns the negative log posterior"
    return -log_posterior(
        state=state,
        x=x,
        y=y,
        iterative=iterative,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        lanczos_key=lanczos_key,
    )


def neg_log_posterior_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
) -> Array:
    "Returns the negative log posterior"
    return -log_posterior_derivs(
        state=state,
        x=x,
        y=y,
        jacobian=jacobian,
        iterative=iterative,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        lanczos_key=lanczos_key,
    )


def fit(
    state: ModelState, x: ArrayLike, y: ArrayLike, iterative: Optional[bool] = False
) -> ModelState:
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
    fit_func = _fit_iter if iterative else _fit_dense
    c, mu, x_locs = fit_func(
        params=state.params,
        x=x,
        y=y,
        key=state.key,
        n_locs=int(state.n_locs),
        kernel=state.kernel,
        mean_function=state.mean_function,
    )
    state = state.update(
        dict(
            x_train=x,
            y_train=y,
            c=c,
            mu=mu,
            is_fitted=True,
            is_fitted_derivs=False,
            x_locs=x_locs,
        )
    )
    return state


def fit_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    iterative: Optional[bool] = False,
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
    fit_func = _fit_derivs_iter if iterative else _fit_derivs_dense
    c, mu, x_locs, jacobian_locs = fit_func(
        params=state.params,
        x=x,
        y=y,
        key=state.key,
        n_locs=int(state.n_locs),
        jacobian=jacobian,
        kernel=state.kernel,
        mean_function=zero_mean,
    )
    state = state.update(
        dict(
            x_train=x,
            y_train=y,
            jacobian_train=jacobian,
            c=c,
            mu=mu,
            is_fitted=False,
            is_fitted_derivs=True,
            x_locs=x_locs,
            jacobian_locs=jacobian_locs,
        )
    )
    return state


def predict(
    state: ModelState,
    x: ArrayLike,
    full_covariance: Optional[bool] = False,
    iterative: Optional[bool] = False,
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
    if state.is_fitted_derivs:
        raise RuntimeError(
            "Model is trained on derivatives. For the prediction,"
            " run `predict_derivs` and `predict_y_derivs`"
        )
    if full_covariance and iterative:
        raise RuntimeError(
            "'full_covariance=True' is not compatible with 'iterative=True'"
        )
    if iterative:
        return _predict_iter(
            params=state.params,
            x_locs=state.x_locs,
            x=x,
            c=state.c,
            mu=state.mu,
            kernel=state.kernel,
        )
    return _predict_dense(
        params=state.params,
        x_locs=state.x_locs,
        x=x,
        c=state.c,
        mu=state.mu,
        kernel=state.kernel,
        full_covariance=full_covariance,
    )


def predict_derivs(
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
    full_covariance: Optional[bool] = False,
    iterative: Optional[bool] = False,
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
    if not state.is_fitted_derivs:
        raise RuntimeError(
            "Model is not fitted. Run `fit_derivs` to fit the model before prediction."
        )
    if state.is_fitted:
        raise RuntimeError(
            "Model is not trained on derivatives. Run `predict` to predict the target."
        )
    if full_covariance and iterative:
        raise RuntimeError(
            "'full_covariance=True' is not compatible with 'iterative=True'"
        )
    if iterative:
        return _predict_derivs_iter(
            params=state.params,
            x_locs=state.x_locs,
            jacobian_locs=state.jacobian_locs,
            x=x,
            jacobian=jacobian,
            c=state.c,
            mu=0.0,
            kernel=state.kernel,
        )
    return _predict_derivs_dense(
        params=state.params,
        x_locs=state.x_locs,
        jacobian_locs=state.jacobian_locs,
        x=x,
        jacobian=jacobian,
        c=state.c,
        mu=0.0,
        kernel=state.kernel,
        full_covariance=full_covariance,
    )


def predict_y_derivs(
    state: ModelState,
    x: ArrayLike,
    full_covariance: Optional[bool] = False,
    iterative: Optional[bool] = False,
) -> Array:
    """predicts with sparse gaussian process
       when the model is trained on derivatives

        μ = K_nm (K_mm + σ²)⁻¹y
        C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        full_covariance: whether to return the covariance matrix too
        iterative: whether to fit iteratively
                   (e.g., never instantiating the kernel)
    Returns:
        μ: predicted mean
        C_nn: predicted covariance
    """
    if not state.is_fitted_derivs:
        raise RuntimeError(
            "Model is not fitted. Run `fit_derivs` to fit the model before prediction."
        )
    if state.is_fitted:
        raise RuntimeError(
            "Model is not trained on derivatives. Run `predict` to predict the target."
        )
    if full_covariance and iterative:
        raise RuntimeError(
            "'full_covariance=True' is not compatible with 'iterative=True'"
        )
    if iterative:
        return _predict_y_derivs_iter(
            params=state.params,
            x_locs=state.x_locs,
            jacobian_locs=state.jacobian_locs,
            x=x,
            c=state.c,
            mu=state.mu,
            kernel=state.kernel,
        )
    return _predict_y_derivs_dense(
        params=state.params,
        x_locs=state.x_locs,
        jacobian_locs=state.jacobian_locs,
        x=x,
        c=state.c,
        mu=state.mu,
        kernel=state.kernel,
        full_covariance=full_covariance,
    )


def sample_prior(
    key: KeyArray,
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


def sample_prior_derivs(
    key: KeyArray,
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
    cov = cov + sigma * jnp.eye(cov.shape[0]) + 1e-10 * jnp.eye(cov.shape[0])

    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def sample_posterior(
    key: KeyArray,
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
    key: KeyArray,
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
    if not state.is_fitted_derivs:
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
    n_locs: ArrayLike,
    key: KeyArray = None,
    mean_function: Callable = data_mean,
    kernel_params: Dict[str, Parameter] = None,
    sigma: Parameter = None,
    loss_fn: Callable = neg_log_marginal_likelihood,
) -> ModelState:
    """initializes the model state of a SGPR RPCholesky-accelerated

    Args:
        kernel: kernel function
        n_locs: number of landmarks (support) points of the SGPR
        key: random PRNG key to perform the randomly pivoted cholesky
        kernel_params: kernel parameters
        sigma: standard deviation of the gaussian noise
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

    if key is None:
        key = jax.random.PRNGKey(time.time())

    params = {"kernel_params": kernel_params, "sigma": sigma}
    opt = {
        "loss_fn": loss_fn,
        "is_fitted": False,
        "is_fitted_derivs": False,
        "c": None,
        "mu": None,
        "n_locs": n_locs,
        "key": key,
    }

    return ModelState(kernel, mean_function, params, **opt)


# =============================================================================
# Sparse Gaussian Process Regression: interface
# =============================================================================


class SGPR_RPChol(BaseGP):
    _init_default = dict(is_fitted=False, is_fitted_derivs=False, c=None, y_mean=None)

    def __init__(
        self,
        kernel: Callable,
        n_locs: int,
        key: KeyArray,
        mean_function: Callable = data_mean,
        kernel_params: Dict[str, Parameter] = None,
        sigma: Parameter = None,
        loss_fn: Callable = neg_log_marginal_likelihood,
    ) -> None:
        """
        Args:
            kernel: kernel function
            n_locs: number of landmark points to use
            key: random PRNG key to perform the RPCholesky
            mean_function: mean function
            kernel_params: kernel parameters
            sigma: standard deviation of the gaussian noise
            loss_fn: loss function

        Note:
            It is always required to specify the loss function
            in the definition of the model for training on
            derivatives of the kernel!
        """
        self.state = init(
            kernel=kernel,
            n_locs=n_locs,
            key=key,
            mean_function=mean_function,
            kernel_params=kernel_params,
            sigma=sigma,
            loss_fn=loss_fn,
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
    _predict_y_derivs_fun = staticmethod(predict_y_derivs)
    # sample policies
    _sample_prior_fun = staticmethod(sample_prior)
    _sample_posterior_fun = staticmethod(sample_posterior)
    _sample_prior_derivs_fun = staticmethod(sample_prior_derivs)
    _sample_posterior_derivs_fun = staticmethod(sample_posterior_derivs)
