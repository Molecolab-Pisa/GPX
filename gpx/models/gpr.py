from __future__ import annotations

from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..bijectors import Softplus
from ..defaults import gpxargs
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
    _mse,
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
# Standard Gaussian Process Regression: functions
# =============================================================================

# Functions to compute the log marginal likelihood, priors, and posteriors


def log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    lanczos_key: Optional[KeyArray] = gpxargs.lanczos_key,
) -> Array:
    """computes the log marginal likelihood for standard gaussian process

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)

    Args:
        state: model state
        x: observations
        y: labels
        iterative: whether to compute the lml iteratively
                   (e.g., never instantiating the kernel)
        num_evals: number of monte carlo evaluations for estimating
                   log|K| (used only if iterative=True)
        num_lanczos: number of Lanczos evaluations for estimating
                     log|K| (used only if iterative=True)
        lanczos_key: random key for Lanczos tridiagonalization
    Returns:
        lml: log marginal likelihood
    """
    if iterative:
        return _lml_iter(
            params=state.params,
            x=x,
            y=y,
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
    """computes the log marginal likelihood for standard gaussian process
    using the Hessian kernel

        lml = - ½ y^T (∂₁∂₂K_nn + σ²I)⁻¹ y - ½ log |∂₁∂₂K_nn + σ²I| - ½ n log(2π)

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
        iterative: whether to compute the lml iteratively
                   (e.g., never instantiating the kernel)
        num_evals: number of monte carlo evaluations for estimating
                   log|K| (used only if iterative=True)
        num_lanczos: number of Lanczos evaluations for estimating
                     log|K| (used only if iterative=True)
        lanczos_key: random key for Lanczos tridiagonalization
    Returns:
        lml: log marginal likelihood
    """
    if iterative:
        return _lml_derivs_iter(
            params=state.params,
            x=x,
            jacobian=jacobian,
            y=y,
            kernel=state.kernel,
            mean_function=zero_mean,
            num_evals=num_evals,
            num_lanczos=num_lanczos,
            lanczos_key=lanczos_key,
        )
    return _lml_derivs_dense(
        params=state.params,
        x=x,
        jacobian=jacobian,
        y=y,
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
        lanczos_key=lanczos_key,
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


def mse_loss(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    y_derivs: ArrayLike,
    coeff: ArrayLike = 1.0,
) -> Array:
    """Computes the mean squared loss for the target and its derivative

        mse = 1/N * Σ_i(μ_i - y_i)**2
             + coeff * 1/N * 1/Natoms * Σ_j||∂μ_j/∂x - ∂y_j/∂x||**2

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
        y_derivs: derivatives of y
        coeff: coefficient to scale forces error
    Returns:
        loss: mean squared loss
    """
    return _mse(
        params=state.params,
        x=x,
        y=y,
        jacobian=jacobian,
        y_derivs=y_derivs,
        kernel=state.kernel,
        mean_function=state.mean_function,
        coeff=coeff,
    )


# Functions to fit a GPR


def fit(
    state: ModelState, x: ArrayLike, y: ArrayLike, iterative: Optional[bool] = False
) -> ModelState:
    """fits a standard gaussian process

        μ = m(y)
        c = (K_nn + σ²I)⁻¹y

    Args:
        state: model state
        x: observations
        y: labels
        iterative: whether to fit iteratively
                   (e.g., never instantiating the kernel)
    Returns:
        state: fitted model state
    """
    fit_func = _fit_iter if iterative else _fit_dense
    c, mu = fit_func(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        mean_function=state.mean_function,
    )
    state = state.update(
        dict(x_train=x, y_train=y, c=c, mu=mu, is_fitted=True, is_fitted_derivs=False)
    )
    return state


def fit_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    iterative: Optional[bool] = False,
) -> ModelState:
    """fits a standard gaussian process

        μ = 0.
        c = (∂₁∂₂K_nn + σ²I)⁻¹y

    Args:
        state: model state
        x: observations
        y: labels
        jacobian: jacobian of x
        iterative: whether to fit iteratively
                   (e.g., never instantiating the kernel)
    Returns:
        state: fitted model state
    """
    fit_func = _fit_derivs_iter if iterative else _fit_derivs_dense
    c, mu = fit_func(
        params=state.params,
        x=x,
        jacobian=jacobian,
        y=y,
        kernel=state.kernel,
        mean_function=zero_mean,  # zero mean
    )
    # also store the contracted jacobian for faster predictions
    ns, _, nv = jacobian.shape
    jaccoef = jnp.einsum("sv,sfv->sf", c.reshape(ns, nv), jacobian)
    state = state.update(
        dict(
            x_train=x,
            y_train=y,
            jacobian_train=jacobian,
            jaccoef=jaccoef,
            c=c,
            mu=mu,
            is_fitted=False,
            is_fitted_derivs=True,
        )
    )
    return state


# Functions to predict with GPR


def predict(
    state: ModelState,
    x: ArrayLike,
    full_covariance: Optional[bool] = False,
    iterative: Optional[bool] = False,
) -> Array:
    """predicts with standard gaussian process

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
    if not state.is_fitted:
        raise RuntimeError(
            "Model is not fitted. Run `fit` to fit the model before prediction."
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
            x_train=state.x_train,
            x=x,
            c=state.c,
            mu=state.mu,
            kernel=state.kernel,
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


def predict_derivs(
    state: ModelState,
    x: ArrayLike,
    jacobian: ArrayLike,
    full_covariance: Optional[bool] = False,
    iterative: Optional[bool] = False,
) -> Array:
    """predicts with standard gaussian process

        μ = K_nm (K_mm + σ²)⁻¹y
        C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

    where K = ∂₁∂₂K

    Args:
        state: model state
        x: observations
        jacobian: jacobian of x
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
    # if full covariance, we can't use the contracted jacobian
    jaccoef = None if full_covariance else state.jaccoef
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
        jaccoef=jaccoef,
    )


def predict_y_derivs(
    state: ModelState,
    x: ArrayLike,
    full_covariance: Optional[bool] = False,
    iterative: Optional[bool] = False,
) -> Array:
    """predicts with standard gaussian process
       when the model is trained on derivatives

        μ = K_nm (K_mm + σ²)⁻¹y
        C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

    Args:
        state: model state
        x: observations
        jacobian: jacobian of x
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
            x_train=state.x_train,
            jacobian_train=state.jacobian_train,
            x=x,
            c=state.c,
            mu=state.mu,
            kernel=state.kernel,
        )
    # if full_covariance, we can't use the contracted jacobian
    jaccoef = None if full_covariance else state.jaccoef
    return _predict_y_derivs_dense(
        params=state.params,
        x_train=state.x_train,
        jacobian_train=state.jacobian_train,
        x=x,
        c=state.c,
        mu=state.mu,
        kernel=state.kernel,
        full_covariance=full_covariance,
        jaccoef=jaccoef,
    )


# TODO Edo: make it accept y and compute the real prior mean
def sample_prior(
    key: KeyArray,
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
    key: KeyArray,
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
    key: KeyArray,
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
    key: KeyArray,
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
    if not state.is_fitted_derivs:
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
        "is_fitted_derivs": False,
        "c": None,
        "mu": None,
    }

    return ModelState(kernel, mean_function, params, **opt)


# =============================================================================
# Standard Gaussian Process Regression: interface
# =============================================================================


class GPR(BaseGP):
    _init_default = dict(is_fitted=False, is_fitted_derivs=False, c=None, mu=None)

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
    _lml_fun = staticmethod(log_marginal_likelihood)
    _lml_derivs_fun = staticmethod(log_marginal_likelihood_derivs)
    _mse_fun = staticmethod(mse_loss)
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
