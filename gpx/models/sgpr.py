from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit
from jax._src import prng
from jax.typing import ArrayLike
from typing_extensions import Self

from ..kernels.operations import kernel_center, kernel_center_test_test
from ..mean_functions import data_mean
from ..optimizers import scipy_minimize
from ..parameters.model_state import ModelState
from ..parameters.parameter import Parameter
from ..priors import NormalPrior
from ..utils import inverse_softplus, softplus
from .utils import (
    _check_object_is_callable,
    _check_object_is_type,
    _check_recursive_dict_type,
    randomized_minimization,
    sample,
)

# =============================================================================
# Sparse Gaussian Process Regression: functions
# =============================================================================


@partial(jit, static_argnums=[3, 4, 5, 6])
def _log_marginal_likelihood(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
    return_negative: Optional[bool] = False,
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
        mean_function=state.mean_function,
        return_negative=return_negative,
        center_kernel=state.center_kernel,
    )


train_loss = partial(log_marginal_likelihood, return_negative=True)


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
    state = state.update(dict(c=c, mu=mu, k_mean=k_mean, is_fitted=True))
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


def default_params() -> Dict[str, Parameter]:
    sigma = Parameter(
        value=1.0,
        trainable=True,
        forward_transform=softplus,
        backward_transform=inverse_softplus,
        prior=NormalPrior(loc=0.0, scale=1.0),
    )

    return dict(sigma=sigma)


def init(
    kernel: Callable,
    mean_function: Callable = data_mean,
    x_locs: Parameter = None,
    kernel_params: Dict[str, Parameter] = None,
    sigma: Parameter = None,
    loss_fn: Callable = train_loss,
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


class SparseGaussianProcessRegression:
    _init_default = dict(is_fitted=False, c=None, y_mean=None, k_mean=None)

    def __init__(
        self,
        kernel: Callable,
        mean_function: Callable = data_mean,
        x_locs: Parameter = None,
        kernel_params: Dict[str, Parameter] = None,
        sigma: Parameter = None,
        loss_fn: Callable = train_loss,
        center_kernel: bool = False,
    ) -> None:
        """
        Args:
            kernel: kernel function
            kernel_params: kernel parameters
            sigma: standard deviation of the gaussian noise
            x_locs: landmark points (support points) of SGPR
            loss_fn: loss function
            center_kernel: whether to center in feature space
        """
        self.state = init(
            kernel=kernel,
            mean_function=mean_function,
            kernel_params=kernel_params,
            sigma=sigma,
            x_locs=x_locs,
            center_kernel=center_kernel,
        )

    @classmethod
    def from_state(cls, state: ModelState) -> "SparseGaussianProcessRegression":
        self = cls.__new__(cls)
        self.state = state
        return self

    def init(
        self,
        kernel: Callable,
        mean_function: Callable = data_mean,
        x_locs: Parameter = None,
        kernel_params: Dict[str, Parameter] = None,
        sigma: Parameter = None,
        loss_fn: Callable = train_loss,
        center_kernel: bool = False,
    ) -> ModelState:
        "resets model state"
        return init(
            kernel=kernel,
            mean_function=mean_function,
            kernel_params=kernel_params,
            sigma=sigma,
            x_locs=x_locs,
            loss_fn=loss_fn,
            center_kernel=center_kernel,
        )

    def default_params(self) -> Dict[str, Parameter]:
        "default model parameters"
        return default_params()

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
        self,
        x: ArrayLike,
        y: ArrayLike,
        minimize_lml: Optional[bool] = True,
        num_restarts: Optional[int] = 0,
        key: prng.PRNGKeyArray = None,
        return_history: Optional[bool] = False,
    ) -> Self:
        """fits a SGPR (projected processes)

            μ = m(y)

            c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn (y - mu)

        Args:
            x: observations
            y: labels
            minimize_lml: whether to tune the parameters to optimize the
                          log marginal likelihood
            num_restarts: number of restarts with randomization to do.
                          If 0, the model is fitted once without any randomization.

        Notes:

        (1) m(y) is the mean function of the real distribution of data. By default,
            we don't make assumptions on the mean of the prior distribution, so it
            is set to the mean value of the input y:

                 μ = (1/n) Σ_i y_i.

        (2) Randomized_minimization requires to optimize the log marginal likelihood.
            In order to optimize with randomized restarts you need to provide a valid
            JAX PRNGKey.
        """
        if minimize_lml:
            minimization_function = scipy_minimize
            self.state, optres, *history = randomized_minimization(
                key=key,
                state=self.state,
                x=x,
                y=y,
                minimization_function=minimization_function,
                num_restarts=num_restarts,
                return_history=return_history,
            )
            self.optimize_results_ = optres

        self.state = fit(self.state, x=x, y=y)

        self.c_ = self.state.c
        self.mu_ = self.state.mu
        self.x_locs_ = self.state.params["x_locs"].value
        self.x_train = x
        self.y_train = y
        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

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

    def save(self, state_file: str) -> Dict:
        """saves the model state values to file"""
        return self.state.save(state_file)

    def load(self, state_file: str) -> Self:
        """loads the model state values from file"""
        self.state = self.state.load(state_file)
        return self

    def randomize(self, key: prng.PRNGKeyArray, reset: Optional[bool] = True) -> Self:
        """Creates a new model state with randomized parameter values"""
        if reset:
            new_state = self.state.randomize(key, opt=self._init_default)
        else:
            new_state = self.state.randomize(key)

        return self.from_state(new_state)


# Alias
SGPR = SparseGaussianProcessRegression
