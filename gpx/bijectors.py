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


@jit
def softmin(x: ArrayLike, minval: ArrayLike) -> Array:
    """softmin transformation

    Computes the softmin transformation of input x.
    Useful to constrain x to be bigger than minval.

    Args:
        x: input array
        minval: lower boundary
    Returns:
        y: transformed array
    """
    return softplus(x - minval) + minval


@jit
def inverse_softmin(x: ArrayLike, minval: ArrayLike) -> Array:
    """inverse softmin transformation

    Computes the inverse of the softmin transformation

    Args:
        x: input array
        minval: lower boundary
    Returns:
        y: transformed array
    """
    return inverse_softplus(x - minval) + minval


@jit
def softmax(x: ArrayLike, maxval: ArrayLike) -> Array:
    """softmax transformation

    Copmutes the softmax transformation of input x.
    Useful to constrain x to be smaller than maxval.

    Args:
        x: input array
        maxval: upper boundary
    Returns:
        y: transformed array
    """
    return -softplus(maxval - x) + maxval


@jit
def inverse_softmax(x: ArrayLike, maxval: ArrayLike) -> Array:
    """inverse softmax transformation

    Computes the inverse of the softmax transformation

    Args:
        x: input array
        maxval: upper boundary
    Returns:
        y: transformed array
    """
    return -inverse_softplus(maxval - x) + maxval


@jit
def softminmax(x: ArrayLike, minval: ArrayLike, maxval: ArrayLike) -> Array:
    """softminmax transformation

    Copmutes the softminmax transformation of input x.
    Useful to constrain x to be greater than minval and
    smaller than maxval.

    Args:
        x: input array
        maxval: upper boundary
    Returns:
        y: transformed array
    """
    return (
        -softplus(maxval - minval - softplus(x - minval))
        * (maxval - minval)
        / (softplus(maxval - minval))
        + maxval
    )


@jit
def inverse_softminmax(x: ArrayLike, minval: ArrayLike, maxval: ArrayLike) -> Array:
    """inverse softminmax transformation

    Computes the inverse of the softminmax transformation

    Args:
        x: input array
        maxval: upper boundary
    Returns:
        y: transformed array
    """
    width = maxval - minval
    return minval + inverse_softplus(
        width - inverse_softplus(softplus(width) * ((maxval - x) / width))
    )


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
        return f"{self.__class__.__name__}()"

    def forward(self, x: ArrayLike) -> Array:
        return softplus(x)

    def backward(self, x: ArrayLike) -> Array:
        return inverse_softplus(x)


class SoftMin:
    """softmin bijector

    This bijector ensures that the argument is bounded below.
    """

    def __init__(self, minval):
        self.minval = minval

    def __str__(self):
        return f"SoftMin({self.minval})"

    def __repr__(self):
        return f"{self.__class__.__name__}(minval={self.minval})"

    def forward(self, x: ArrayLike) -> Array:
        return softmin(x, minval=self.minval)

    def backward(self, x: ArrayLike) -> Array:
        return inverse_softmin(x, minval=self.minval)


class SoftMax:
    """softmax bijector

    This bijector ensures that the argument is bounded above.
    """

    def __init__(self, maxval):
        self.maxval = maxval

    def __str__(self):
        return f"SoftMax({self.maxval})"

    def __repr__(self):
        return f"{self.__class__.__name__}(maxval={self.maxval})"

    def forward(self, x: ArrayLike) -> Array:
        return softmax(x, maxval=self.maxval)

    def backward(self, x: ArrayLike) -> Array:
        return inverse_softmax(x, maxval=self.maxval)


class SoftMinMax:
    """softminmax bijector

    This bijector ensures that the argument is bounded below and above.
    """

    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def __str__(self):
        return f"SoftMinMax({self.minval}, {self.maxval})"

    def __repr__(self):
        return f"{self.__class__.__name__}(minval={self.minval}, maxval={self.maxval})"

    def forward(self, x: ArrayLike) -> Array:
        return softminmax(x, minval=self.minval, maxval=self.maxval)

    def backward(self, x: ArrayLike) -> Array:
        return inverse_softminmax(x, minval=self.minval, maxval=self.maxval)


class Identity:
    """identity bijector

    The do-nothing bijector
    """

    def __init__(self):
        pass

    def __str__(self):
        return "Identity"

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, x: Any) -> Any:
        return x

    def backward(self, x: Any) -> Any:
        return x
