from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

# Functions


@jit
def softplus(x: ArrayLike) -> Array:
    """softplus transformation

    Computes the softplus transformation of the input x

        y = log(1 + exp(x))

    Useful to constrain x to positive values.

    Note: a small jitter is added to prevent NaNs when
          inverting this function with inverse_softplus
          This means that inputs below ~ -690 will not
          be recovered exactly by the inverse function

    Args:
        x: input array
    Returns:
        y: transformed array
    """
    return jnp.logaddexp(x, 0.0) + 1e-300


@jit
def inverse_softplus(x: ArrayLike) -> Array:
    """inverse softplus transformation

    Computes the inverse of the softplus transformation
    using equation (2) below

        y = log(exp(x) - 1)                           (1)
          = log(1 - exp(-x)) + x                      (2)

    This function is stable when x is large and positive,
    compared to the naive inverse (1).
    It is not stable when x is really small (< ~1e-300),
    which can happen when x is the output of softplus
    applied to a large and negative number.

    Args:
        x: input array
    Returns:
        y: transformed array
    """
    return x + jnp.log(-jnp.expm1(-x))


# Classes


class Softplus:
    """softplus bijector

    This bijector ensures positivity of the argument.
    """

    def __init__(self):
        pass

    def __str__(self):
        return "Softplus"

    def __repr__(self):
        return "{self.__class__.__name__}()"

    def forward(self, x: ArrayLike) -> Array:
        return softplus(x)

    def backward(self, x: ArrayLike) -> Array:
        return inverse_softplus(x)


class Identity:
    """identity bijector

    The do-nothing bijector
    """

    def __init__(self):
        pass

    def __str__(self):
        return "Identity"

    def __repr__(self):
        return "{self.__class__.__name__}()"

    def forward(self, x: Any) -> Any:
        return x

    def backward(self, x: Any) -> Any:
        return x
