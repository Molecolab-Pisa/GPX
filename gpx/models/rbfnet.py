from __future__ import annotations
from typing import Callable, Dict, Tuple

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import random
from jax._src import prng

from ..utils import identity
from ..parameters import ModelState
from ..parameters.parameter import parse_param, Parameter


@partial(jit, static_argnums=[3])
def _train_loss(
    params: Dict[str, Parameter], x: jnp.ndarray, y: jnp.ndarray, kernel: Callable
) -> jnp.ndarray:
    y_pred = _predict(params=params, x_train=x, x_pred=x, kernel=kernel)
    return jnp.mean((y_pred - y) ** 2)


def train_loss(state: ModelState, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return _train_loss(params=state.params, x=x, y=y, kernel=state.kernel)


@partial(jit, static_argnums=[3])
def _predict(
    params: Dict[str, Parameter],
    x_train: jnp.ndarray,
    x_pred: jnp.ndarray,
    kernel: Callable,
) -> jnp.ndarray:
    kernel_params = params["kernel_params"]
    weights = params["weights"].value

    gram = kernel(x_pred, x_train, kernel_params)
    pred = jnp.dot(gram, weights)

    return pred


def predict(
    state: ModelState, x_train: jnp.ndarray, x_pred: jnp.ndarray
) -> jnp.ndarray:
    return _predict(
        params=state.params, x_train=x_train, x_pred=x_pred, kernel=state.kernel
    )


def init(
    key: prng.PRNGKeyArray,
    kernel: Callable,
    kernel_params: Dict[str, Tuple],
    num_input: int,
    num_output: int,
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

    params = {"kernel_params": kp, "weights": weights}

    return ModelState(kernel, params)


# =============================================================================
# RBF Network: interface
# =============================================================================


class RadialBasisFunctionNetwork:
    def __init__(self):
        raise NotImplementedError


# Alias
RBFNet = RadialBasisFunctionNetwork
