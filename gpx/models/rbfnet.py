from __future__ import annotations
from typing import Callable, Dict, Tuple

from jax import random
from jax._src import prng

from ..utils import identity
from ..parameters import ModelState
from ..parameters.parameter import parse_param, Parameter


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
