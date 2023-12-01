from __future__ import annotations

from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax._src import prng
from jax.typing import ArrayLike

from ..bijectors import Softplus
from ..mean_functions import data_mean, zero_mean
from ..parameters import ModelState
from ..parameters.parameter import Parameter, is_parameter
from ..priors import GammaPrior
from ._gpr import (
    _A_derivs_lhs,
    _A_lhs,
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
)
from .base import BaseGP
from .utils import (
    _check_object_is_callable,
    _check_object_is_type,
    _check_recursive_dict_type,
    sample,
)

# =============================================================================
# Standard Gaussian Process Regression: functions
# =============================================================================

# Functions to compute the log marginal likelihood, priors, and posteriors


def log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
) -> Array:
    """computes the log marginal likelihood for standard gaussian process

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)

    Args:
        state: model state
        x: observations
        y: labels
    Returns:
        lml: log marginal likelihood
    """
    return _lml_dense(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        mean_function=state.mean_function,
    )


def log_marginal_likelihood_iter(state, x, y, num_evals, num_lanczos, lanczos_key):
    return _lml_iter(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        mean_function=state.mean_function,
        num_evals=int(num_evals),
        num_lanczos=int(num_lanczos),
        lanczos_key=lanczos_key,
    )


def log_marginal_likelihood_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
) -> Array:
    """computes the log marginal likelihood for standard gaussian process

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
    Returns:
        lml: log marginal likelihood
    """
    return _lml_derivs_dense(
        params=state.params,
        x=x,
        jacobian=jacobian,
        y=y,
        kernel=state.kernel,
        mean_function=zero_mean,
    )


def log_marginal_likelihood_derivs_iter(
    state,
    x,
    y,
    jacobian,
    num_evals,
    num_lanczos,
    lanczos_key,
):
    return _lml_derivs_iter(
        params=state.params,
        x=x,
        jacobian=jacobian,
        y=y,
        kernel=state.kernel,
        mean_function=zero_mean,
        num_evals=int(num_evals),
        num_lanczos=int(num_lanczos),
        lanczos_key=lanczos_key,
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


def neg_log_marginal_likelihood_iter(state, x, y, num_evals, num_lanczos, lanczos_key):
    return -log_marginal_likelihood_iter(
        state, x, y, num_evals, num_lanczos, lanczos_key
    )


def neg_log_marginal_likelihood_derivs(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> Array:
    "Returns the negative log marginal likelihood"
    return -log_marginal_likelihood_derivs(state=state, x=x, y=y, jacobian=jacobian)


def neg_log_marginal_likelihood_derivs_iter(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    num_evals,
    num_lanczos,
    lanczos_key,
) -> Array:
    "Returns the negative log marginal likelihood"
    return -log_marginal_likelihood_derivs_iter(
        state=state,
        x=x,
        y=y,
        jacobian=jacobian,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        lanczos_key=lanczos_key,
    )


def neg_log_posterior(state: ModelState, x: ArrayLike, y: ArrayLike) -> Array:
    "Returns the negative log posterior"
    return -log_posterior(state=state, x=x, y=y)


def neg_log_posterior_derivs(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> Array:
    "Returns the negative log posterior"
    return -log_posterior_derivs(state=state, x=x, y=y, jacobian=jacobian)


# Functions to fit a GPR


def fit(state: ModelState, x: ArrayLike, y: ArrayLike) -> ModelState:
    """fits a standard gaussian process

        μ = m(y)
        c = (K_nn + σ²I)⁻¹y

    Args:
        state: model state
        x: observations
        y: labels
    Returns:
        state: fitted model state
    """
    c, mu = _fit_dense(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        mean_function=state.mean_function,
    )
    state = state.update(dict(x_train=x, y_train=y, c=c, mu=mu, is_fitted=True))
    return state


def fit_iter(state: ModelState, x: ArrayLike, y: ArrayLike) -> ModelState:
    """fits a standard GPR iteratively

    Fits a standard GPR solving the linear system iteratively
    with Conjugated Gradient.

    μ = m(y)
    c = (K(x, x) + σ²I)⁻¹y

    Args:
        state: model state
        x: observations
        y: labels
    Returns:
        state: fitted model state
    """
    c, mu = _fit_iter(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        mean_function=state.mean_function,
    )
    state = state.update(dict(x_train=x, y_train=y, c=c, mu=mu, is_fitted=True))
    return state


def fit_derivs(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> ModelState:
    """fits a standard gaussian process

        μ = 0.
        c = (∂∂K_nn + σ²I)⁻¹y

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
    Returns:
        state: fitted model state
    """
    c, mu = _fit_derivs_dense(
        params=state.params,
        x=x,
        jacobian=jacobian,
        y=y,
        kernel=state.kernel,
        mean_function=zero_mean,  # zero mean
    )
    state = state.update(
        dict(
            x_train=x,
            y_train=y,
            jacobian_train=jacobian,
            c=c,
            mu=mu,
            is_fitted=True,
        )
    )
    return state


def fit_derivs_iter(
    state: ModelState, x: ArrayLike, y: ArrayLike, jacobian: ArrayLike
) -> ModelState:
    """fits a standard GPR iteratively when training on derivatives

    Fits a standard GPR solving the linear system iteratively with
    Conjugate Gradient when training on derivative values.

    μ = 0.
    c = (∂∂K(x, x) + σ²I)⁻¹y

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
    Returns:
        state: fitted model state
    """
    c, mu = _fit_derivs_iter(
        params=state.params,
        x=x,
        jacobian=jacobian,
        y=y,
        kernel=state.kernel,
        mean_function=zero_mean,  # zero mean
    )
    state = state.update(
        dict(
            x_train=x,
            y_train=y,
            jacobian_train=jacobian,
            c=c,
            mu=mu,
            is_fitted=True,
        )
    )
    return state


# Functions to predict with GPR


def predict(
    state: ModelState,
    x: ArrayLike,
    full_covariance: Optional[bool] = False,
) -> Array:
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
    return _predict_dense(
        params=state.params,
        x_train=state.x_train,
        x=x,
        c=state.c,
        mu=state.mu,
        kernel=state.kernel,
        full_covariance=full_covariance,
    )


def predict_iter(
    state: ModelState,
    x: ArrayLike,
) -> Array:
    """predicts with GPR

    Predicts with GPR without instantiating the full matrix.
    The contraction with the linear coefficients is performed
    iteratively.

    Args:
        state: model state
        x: observations
    Returns:
        μ: predicted mean
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Model is not fitted. Run `fit` to fit the model before prediction."
        )
    return _predict_iter(
        params=state.params,
        x_train=state.x_train,
        x=x,
        c=state.c,
        mu=state.mu,
        kernel=state.kernel,
    )


def predict_derivs(
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
    full_covariance: Optional[bool] = False,
) -> Array:
    """predicts with standard gaussian process

        μ = K_nm (K_mm + σ²)⁻¹y
        C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        jacobian: jacobian of x
        full_covariance: whether to return the covariance matrix too
    Returns:
        μ: predicted mean
        C_nn: predicted covariance
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Model is not fitted. Run `fit` to fit the model before prediction."
        )
    return _predict_derivs_dense(
        params=state.params,
        x_train=state.x_train,
        jacobian_train=state.jacobian_train,
        x=x,
        jacobian=jacobian,
        c=state.c,
        mu=0.0,  # zero mean
        kernel=state.kernel,
        full_covariance=full_covariance,
    )


def predict_derivs_iter(
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
) -> Array:
    """predicts derivative values with GPR

    Predicts the derivative values with GPR.
    The contraction with the linear coefficients is performed
    iteratively.

    Args:
        state: model state
        x: observations
        jacobian: jacobian of x
    Returns:
        μ: predicted mean
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Model is not fitted. Run `fit` to fit the model before prediction."
        )
    return _predict_derivs_iter(
        params=state.params,
        x_train=state.x_train,
        jacobian_train=state.jacobian_train,
        x=x,
        jacobian=jacobian,
        c=state.c,
        mu=0.0,
        kernel=state.kernel,
    )


# TODO Edo: make it accept y and compute the real prior mean
def sample_prior(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    """returns samples from the prior of a gaussian process

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        n_samples: number of samples to draw

    Returns:
        samples: samples from the prior distribution
    """
    mean = jnp.zeros(x.shape)
    cov = _A_lhs(x, x, state.params, state.kernel, noise=True)
    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def sample_prior_derivs(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    """returns samples from the prior of a gaussian process using the hessian kernel

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        jacobian: jacobian of x
        n_samples: number of samples to draw

    Returns:
        samples: samples from the prior distribution
    """
    mean = jnp.zeros(x.shape)
    cov = _A_derivs_lhs(
        x, jacobian, x, jacobian, state.params, state.kernel, noise=True
    )
    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def sample_posterior(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    """returns samples from the posterior of a gaussian process

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        n_samples: number of samples to draw

    Returns:
        samples: samples from the posterior distribution
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Cannot sample from the posterior if the model is not fitted"
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
    """returns samples from the posterior of a gaussian process

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        jacobian: jacobian of x
        n_samples: number of samples to draw

    Returns:
        samples: samples from the posterior distribution
    """
    if not state.is_fitted:
        raise RuntimeError(
            "Cannot sample from the posterior if the model is not fitted"
        )
    mean, cov = predict_derivs(state, x=x, jacobian=jacobian, full_covariance=True)
    cov += 1e-10 * jnp.eye(cov.shape[0])
    return sample(key=key, mean=mean, cov=cov, n_samples=n_samples)


def default_params() -> Dict[str, Parameter]:
    sigma = Parameter(
        value=1.0,
        trainable=True,
        bijector=Softplus(),
        prior=GammaPrior(),
    )
    return dict(sigma=sigma)


def init(
    kernel: Callable,
    mean_function: Callable = data_mean,
    kernel_params: Dict[str, Parameter] = None,
    sigma: Parameter = None,
    loss_fn: Callable = neg_log_marginal_likelihood,
) -> ModelState:
    """initializes the model state of a gaussian process

    Args:
        kernel: kernel function
        kernel_params: kernel parameters
        sigma: standard deviation of gaussian noise
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
        sigma = default_params()["sigma"]

    else:
        _check_object_is_type(sigma, Parameter, "sigma")

    params = {"kernel_params": kernel_params, "sigma": sigma}
    opt = {
        "x_train": None,
        "y_train": None,
        "loss_fn": loss_fn,
        "is_fitted": False,
        "c": None,
        "mu": None,
    }

    return ModelState(kernel, mean_function, params, **opt)


# =============================================================================
# Standard Gaussian Process Regression: interface
# =============================================================================


class GPR(BaseGP):
    _init_default = dict(is_fitted=False, c=None, mu=None)

    def __init__(
        self,
        kernel: Callable,
        mean_function: Callable = data_mean,
        kernel_params: Dict[str, Parameter] = None,
        sigma: Parameter = None,
        loss_fn: Callable = neg_log_marginal_likelihood,
    ) -> None:
        """
        Args:
            kernel: kernel function
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
            mean_function=mean_function,
            kernel_params=kernel_params,
            sigma=sigma,
            loss_fn=loss_fn,
        )

    # parameters
    _default_params_fun = staticmethod(default_params)
    _init_fun = staticmethod(init)
    # lml
    _lml_dense_fun = staticmethod(log_marginal_likelihood)
    _lml_iter_fun = staticmethod(log_marginal_likelihood_iter)
    _lml_derivs_dense_fun = staticmethod(log_marginal_likelihood_derivs)
    _lml_derivs_iter_fun = staticmethod(log_marginal_likelihood_derivs_iter)
    # fit policies
    _fit_dense_fun = staticmethod(fit)
    _fit_iter_fun = staticmethod(fit_iter)
    _fit_derivs_dense_fun = staticmethod(fit_derivs)
    _fit_derivs_iter_fun = staticmethod(fit_derivs_iter)
    # prediction policies
    _predict_dense_fun = staticmethod(predict)
    _predict_iter_fun = staticmethod(predict_iter)
    _predict_derivs_dense_fun = staticmethod(predict_derivs)
    _predict_derivs_iter_fun = staticmethod(predict_derivs_iter)
    # sample policies
    _sample_prior_fun = staticmethod(sample_prior)
    _sample_posterior_fun = staticmethod(sample_posterior)
    _sample_prior_derivs_fun = staticmethod(sample_prior_derivs)
    _sample_posterior_derivs_fun = staticmethod(sample_posterior_derivs)
