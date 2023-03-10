from __future__ import annotations
from typing import Callable, Dict, Tuple
from typing_extensions import Self

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import random
from jax._src import prng

from ..utils import identity, softplus, inverse_softplus
from ..parameters import ModelState
from ..parameters.parameter import parse_param, Parameter
from ..optimize import scipy_minimize


@partial(jit, static_argnums=[3, 4])
def _train_loss(
    params: Dict[str, Parameter],
    x: jnp.ndarray,
    y: jnp.ndarray,
    kernel: Callable,
    output_layer: Callable,
) -> jnp.ndarray:
    y_pred = _predict(
        params=params, x_train=x, x_pred=x, kernel=kernel, output_layer=output_layer
    )

    alpha = params["alpha"].value
    weights = params["weights"].value
    n_samples = y.shape[0]

    return (
        jnp.mean((y_pred - y) ** 2)
        + alpha * jnp.sum(weights.ravel().T @ weights.ravel()) / n_samples
    )


def train_loss(state: ModelState, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return _train_loss(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        output_layer=state.output_layer,
    )


@partial(jit, static_argnums=[3, 4])
def _predict(
    params: Dict[str, Parameter],
    x_train: jnp.ndarray,
    x_pred: jnp.ndarray,
    kernel: Callable,
    output_layer: Callable,
) -> jnp.ndarray:
    kernel_params = params["kernel_params"]
    weights = params["weights"].value

    gram = kernel(x_pred, x_train, kernel_params)
    pred = jnp.dot(gram, weights)

    pred = output_layer(pred)

    return pred


def predict(
    state: ModelState, x_train: jnp.ndarray, x_pred: jnp.ndarray
) -> jnp.ndarray:
    return _predict(
        params=state.params,
        x_train=x_train,
        x_pred=x_pred,
        kernel=state.kernel,
        output_layer=state.output_layer,
    )


def init(
    key: prng.PRNGKeyArray,
    kernel: Callable,
    kernel_params: Dict[str, Tuple],
    num_input: int,
    num_output: int,
    output_layer: Callable = identity,
    alpha: Tuple = (1.0, True, softplus, inverse_softplus),
    loss_fn: Callable = train_loss,
) -> ModelState:
    if not callable(kernel):
        raise RuntimeError(
            f"kernel must be provided as a callable function, you provided {type(kernel)}"
        )

    if not isinstance(kernel_params, dict):
        raise RuntimeError(
            f"kernel_params must be provided as a dictionary, you provided {type(kernel_params)}"
        )

    weights = Parameter(
        random.normal(key, shape=(num_input, num_output)),
        True,
        identity,
        identity,
    )

    kp = {}
    for key in kernel_params:
        param = kernel_params[key]
        kp[key] = parse_param(param)

    alpha = parse_param(alpha)

    # Careful, as here the order matters (thought it shouldn't for a good api)
    params = {"alpha": alpha, "kernel_params": kp, "weights": weights}
    opt = {"output_layer": output_layer, "loss_fn": loss_fn}

    return ModelState(kernel, params, **opt)


# =============================================================================
# RBF Network: interface
# =============================================================================


class RadialBasisFunctionNetwork:
    def __init__(
        self,
        key: prng.PRNGKeyArray,
        kernel: Callable,
        kernel_params: Dict[str, Tuple],
        num_input: int,
        num_output: int,
        output_layer: Callable = identity,
        alpha: Tuple = (1.0, True, softplus, inverse_softplus),
        loss_fn: Callable = train_loss,
    ) -> None:
        self.key = key
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.num_input = num_input
        self.num_output = num_output
        self.alpha = alpha
        self.output_layer = output_layer
        self.loss_fn = loss_fn
        self.state = init(
            key=key,
            kernel=kernel,
            kernel_params=kernel_params,
            num_input=num_input,
            num_output=num_output,
            alpha=alpha,
            output_layer=output_layer,
            loss_fn=loss_fn,
        )

    def print(self) -> None:
        "prints the model parameters"
        return self.state.print_params()

    def fit(self, x: jnp.ndarray, y: jnp.ndarray) -> Self:
        self.state, optres = scipy_minimize(
            self.state, x=x, y=y, loss_fn=self.state.loss_fn
        )
        self.optimize_results_ = optres
        self.x_train = x

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        if not hasattr(self, "x_train"):
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} is not fitted yet. "
                "Call 'fit' before using this model for prediction."
            )
        return predict(self.state, x_train=self.x_train, x_pred=x)


# Alias
RBFNet = RadialBasisFunctionNetwork
