from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
from jax import Array, jit, random
from jax._src import prng
from jax.typing import ArrayLike
from typing_extensions import Self

from ..bijectors import Identity, Softplus
from ..kernels.operations import kernel_center
from ..optimizers import NLoptWrapper, optax_minimize, scipy_minimize
from ..parameters import ModelState
from ..parameters.parameter import Parameter
from ..priors import NormalPrior
from .utils import (
    _check_object_is_callable,
    _check_object_is_type,
    _check_recursive_dict_type,
    randomized_minimization,
)

# optax optimizer
GradientTransformation = Any


def identity(x: Any) -> Any:
    return x


@partial(jit, static_argnums=[3, 4, 5, 6])
def _train_loss(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    output_layer: Callable,
    center_kernel: bool,
    k_mean: ArrayLike,
) -> Array:
    y_pred, _ = _predict(
        params=params,
        x=x,
        kernel=kernel,
        output_layer=output_layer,
        center_kernel=center_kernel,
        k_mean=k_mean,
    )

    alpha = params["alpha"].value
    weights = params["weights"].value
    n_samples = y.shape[0]

    return (
        jnp.mean((y_pred - y) ** 2)
        + alpha * jnp.sum(weights.ravel().T @ weights.ravel()) / n_samples
    )


def train_loss(state: ModelState, x: ArrayLike, y: ArrayLike) -> Array:
    return _train_loss(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        output_layer=state.output_layer,
        center_kernel=state.center_kernel,
        k_mean=state.k_mean,
    )


@partial(jit, static_argnums=[2, 3])
def _predict_linear(
    params: Dict[str, Parameter],
    x: ArrayLike,
    kernel: Callable,
    center_kernel: bool,
    k_mean: ArrayLike,
) -> Array:
    kernel_params = params["kernel_params"]
    weights = params["weights"].value
    x_inducing = params["inducing_points"].value

    gram = kernel(x_inducing, x, kernel_params)

    if center_kernel:
        if k_mean is None:
            k_mean = jnp.mean(gram, axis=0)
        gram = kernel_center(gram, k_mean)

    pred = jnp.einsum("ij,io->jo", gram, weights)

    return pred, k_mean


@partial(jit, static_argnums=[2, 3, 4])
def _predict(
    params: Dict[str, Parameter],
    x: ArrayLike,
    kernel: Callable,
    output_layer: Callable,
    center_kernel: bool,
    k_mean: ArrayLike,
) -> Array:
    pred, k_mean = _predict_linear(
        params=params, x=x, kernel=kernel, center_kernel=center_kernel, k_mean=k_mean
    )
    pred = output_layer(pred)

    return pred, k_mean


def predict(state: ModelState, x: ArrayLike, linear_only: bool = False) -> Array:
    if linear_only:
        return _predict_linear(
            params=state.params,
            x=x,
            kernel=state.kernel,
            center_kernel=state.center_kernel,
            k_mean=state.k_mean,
        )
    else:
        return _predict(
            params=state.params,
            x=x,
            kernel=state.kernel,
            output_layer=state.output_layer,
            center_kernel=state.center_kernel,
            k_mean=state.k_mean,
        )


def default_params(
    key: prng.PRNGKeyArray, num_input: int, num_output: int
) -> Dict[str, Parameter]:
    # regularization strength
    alpha = Parameter(
        value=1.0,
        trainable=True,
        bijector=Softplus(),
        prior=NormalPrior(loc=0.0, scale=1.0),
    )

    # weights
    weights = Parameter(
        value=random.normal(key, shape=(num_input, num_output)),
        trainable=True,
        bijector=Identity(),
        prior=NormalPrior(loc=0.0, scale=1.0, shape=(num_input, num_output)),
    )

    return dict(alpha=alpha, weights=weights)


def init(
    key: prng.PRNGKeyArray,
    kernel: Callable,
    inducing_points: Parameter,
    num_output: int,
    kernel_params: Dict[str, Parameter] = None,
    alpha: Parameter = None,
    output_layer: Callable = identity,
    loss_fn: Callable = train_loss,
    center_kernel: bool = False,
) -> ModelState:
    # kernel
    _check_object_is_callable(kernel, "kernel")

    # kernel parameters
    if kernel_params is None:
        kernel_params = kernel.default_params()
    else:
        _check_object_is_type(kernel_params, dict, "kernel_params")
        _check_recursive_dict_type(kernel_params, Parameter)

    # inducing points
    _check_object_is_type(inducing_points, Parameter, "inducing_points")

    # number of inducing points
    num_input = inducing_points.value.shape[0]

    _defaults = default_params(key, num_input, num_output)
    weights = _defaults.pop("weights")

    if alpha is None:
        alpha = _defaults.pop("alpha")
    else:
        _check_object_is_type(alpha, Parameter, "alpha")

    # Careful, as here the order matters (thought it shouldn't for a good api)
    params = {
        "alpha": alpha,
        "inducing_points": inducing_points,
        "kernel_params": kernel_params,
        "weights": weights,
    }
    opt = {
        "output_layer": output_layer,
        "loss_fn": loss_fn,
        "center_kernel": center_kernel,
        "k_mean": jnp.zeros(num_input),
    }

    return ModelState(kernel=kernel, mean_function=None, params=params, **opt)


# =============================================================================
# RBF Network: interface
# =============================================================================


class RBFNet:
    "Radial Basis Function Network"

    _init_default = dict(output_layer=identity, loss_fn=train_loss, center_kernel=False)

    def __init__(
        self,
        key: prng.PRNGKeyArray,
        kernel: Callable,
        inducing_points: Parameter,
        num_output: int,
        kernel_params: Dict[str, Parameter] = None,
        alpha: Parameter = None,
        output_layer: Callable = identity,
        loss_fn: Callable = train_loss,
        center_kernel: Optional[bool] = False,
    ) -> None:
        """
        Args:
            key: JAX random PRNGKey.
                 used for instantiating the model weights (they
                 are initialized by sampling a normal distribution)
            kernel: kernel function.
            kernel_params: kernel parameters.
                           should be provided as a dictionary mapping
                           the parameter name to a 4-tuple, which specifies
                           the value, whether the parameter is trainable, and the
                           forward and backward transformation functions for the
                           parameter.
            inducing_points: inducing points.
                             the kernel matrix is evaluated between points provided
                             by the user and the inducing points.
                             if the number of inducing points is smaller than the
                             train set points, then the model is sparse.
                             should be provided as a 4-tuple, as they are casted as
                             a parameter of the model.
            num_output: number of outputs of the RBF layer.
            output_layer: output_layer, taking as input the prediction of the RBF
                          layer, and outputting a transformed prediction.
                          should accept a ArrayLike with shape (n, num_output),
                          where n is the number of samples, and should output another
                          ArrayLike of shape (n, num_output')
            alpha: regularization parameter of the L2 regularization term in the
                   default loss function. If another loss is used, alpha is ignored.
            loss_fn: loss function used to optimize the model parameters.
                     by default, it minimizes the squared error plus the L2
                     regularization term.
            center_kernel: whether to center in feature space
        """
        self.state = init(
            key=key,
            kernel=kernel,
            kernel_params=kernel_params,
            inducing_points=inducing_points,
            num_output=num_output,
            alpha=alpha,
            output_layer=output_layer,
            loss_fn=loss_fn,
            center_kernel=center_kernel,
        )

    @classmethod
    def from_state(cls, state: ModelState) -> "RBFNet":
        self = cls.__new__(cls)
        self.state = state
        return self

    def init(
        self,
        key: prng.PRNGKeyArray,
        kernel: Callable,
        inducing_points: Parameter,
        num_output: int,
        kernel_params: Dict[str, Parameter] = None,
        alpha: Parameter = None,
        output_layer: Callable = identity,
        loss_fn: Callable = train_loss,
        center_kernel: bool = False,
    ) -> ModelState:
        "resets model state"
        return init(
            key=key,
            kernel=kernel,
            kernel_params=kernel_params,
            inducing_points=inducing_points,
            num_output=num_output,
            alpha=alpha,
            output_layer=output_layer,
            loss_fn=loss_fn,
            center_kernel=center_kernel,
        )

    def default_params(
        self, key: prng.PRNGKeyArray, num_input: int, num_output: int
    ) -> Dict[str, Parameter]:
        "default model parameters"
        return default_params(key=key, num_input=num_input, num_output=num_output)

    def print(self) -> None:
        "prints the model parameters"
        return self.state.print_params()

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        num_restarts: Optional[int] = 0,
        key: prng.PRNGKeyArray = None,
        return_history: Optional[bool] = False,
    ) -> Self:
        minimization_function = scipy_minimize
        self.state = self.state.update({"k_mean": None})
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
        self.x_train = x
        self.y_train = y
        _, k_mean = predict(self.state, x=x)
        self.state = self.state.update({"k_mean": k_mean})
        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self

    def fit_nlopt(
        self,
        x: ArrayLike,
        y: ArrayLike,
        opt: NLoptWrapper,
        num_restarts: Optional[int] = 0,
        key: Optional[prng.PRNGKeyArray] = None,
        return_history: Optional[bool] = False,
    ) -> Self:
        minimization_function = opt.optimize
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
        self.x_train = x
        self.y_train = y
        _, k_mean = predict(self.state, x=x)
        self.state = self.state.update({"k_mean": k_mean})
        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self

    def fit_optax(
        self,
        x: ArrayLike,
        y: ArrayLike,
        optimizer: GradientTransformation = None,
        x_val: ArrayLike = None,
        y_val: ArrayLike = None,
        nsteps: int = 10,
        learning_rate: float = 1.0,
        update_every: int = 1,
        num_restarts: int = 0,
        key: prng.PRNGKeyArray = None,
        return_history: int = False,
    ) -> Self:
        minimization_function = optax_minimize
        self.state = self.state.update({"k_mean": None})
        self.state, opt_state, epochs_history, *history = randomized_minimization(
            key=key,
            state=self.state,
            x=x,
            y=y,
            minimization_function=minimization_function,
            num_restarts=num_restarts,
            return_history=return_history,
            opt_kwargs=dict(
                optimizer=optimizer,
                x_val=x_val,
                y_val=y_val,
                nsteps=nsteps,
                learning_rate=learning_rate,
                update_every=update_every,
            ),
        )

        self.optax_opt_state_ = opt_state
        self.optax_history_ = epochs_history
        self.x_train = x
        self.y_train = y
        _, k_mean = predict(self.state, x=x)
        self.state = self.state.update({"k_mean": k_mean})
        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self

    def predict(self, x: ArrayLike, linear_only: bool = False) -> Array:
        pred, _ = predict(self.state, x=x, linear_only=linear_only)
        return pred

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
