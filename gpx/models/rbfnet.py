from __future__ import annotations
from typing import Callable, Dict, Tuple

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import random
from jax._src import prng

from ..utils import identity, softplus, inverse_softplus
from ..parameters import ModelState
from ..parameters.parameter import parse_param, Parameter


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
    opt = {"output_layer": output_layer}

    return ModelState(kernel, params, **opt)


# =============================================================================
# RBF Network: interface
# =============================================================================


class RadialBasisFunctionNetwork:
    def __init__(self):
        raise NotImplementedError


# Alias
RBFNet = RadialBasisFunctionNetwork
