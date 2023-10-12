from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Tuple

import jax.numpy as jnp
from jax import Array, random
from jax._src import prng
from jax.typing import ArrayLike

# ============================================================================
# Functions
# ============================================================================


def normal_sample(
    key: prng.PRNGKeyArray,
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
    shape: Tuple = (),
    dtype: Any = jnp.float64,
) -> Array:
    """samples from a normal distribution

        p(θ) = ( 1/(2πσ²)^½ ) e^( -(θ-μ)²/(2σ²) )

    Args:
        key: PRNGKey
        loc: mean of the distribution
        scale: standard deviation of the distribution
        shape: shape of the sample
        dtype: dtype of the sample

    Returns:
        sample: sample from the normal distribution
    """
    return loc + scale * random.normal(key=key, shape=shape, dtype=dtype)


def normal_logpdf(
    x: ArrayLike,
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
):
    """
    Evaluates the log pdf of the normal distribution

        log p(θ) = - log(σ) - ½log(2π) - (1/(2σ²)) (θ - μ)²

    Args:
        x: input point
        loc: mean of the distribution
        scale: standard deviation of the distribution

    Returns:
        logpdf: log probability density function evaluated at x
    """
    return (
        -jnp.log(scale)
        - 0.5 * jnp.log(2 * jnp.pi)
        - 0.5 * (1.0 / scale**2) * (x - loc) ** 2
    )


# ============================================================================
# Base abstract class
# ============================================================================


class Prior(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        """representation of the prior

        Representation of the object. This should
        output the precise code that creates the instance,
        e.g., copy-paste of this output must create
        the same instance.
        """

    @abstractmethod
    def __str__(self) -> str:
        """string representation of the prior

        This should output a simplified representation
        of the prior. It appears in the table printed when
        calling the `print_params` method of the ModelState
        or the `print` method of the model classes. As such,
        it has to be concise.

        E.g., for the NormalPrior, this could output Normal(0, 1),
        which has an understandable meaning.
        """

    @abstractmethod
    def __copy__(self) -> "Prior":
        "return a shallow copy of the prior"

    @abstractmethod
    def sample(self, key: prng.PRNGKeyArray) -> Array:
        "sample from the prior"

    @abstractmethod
    def logpdf(self, x: ArrayLike) -> Array:
        "evaluate the log pdf"

    def __call__(self, key: prng.PRNGKeyArray) -> Array:
        return self.sample(key)

    def copy(self) -> "Prior":
        return copy.copy(self)


# ============================================================================
# Classes
# ============================================================================


class NormalPrior(Prior):
    def __init__(self, loc=0.0, scale=1.0, shape=(), dtype=jnp.float64):
        self.shape = shape
        self.dtype = dtype
        # initialize after assigning dtype so that
        # the class knows what dtype to use
        self.loc = loc
        self.scale = scale

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value: ArrayLike) -> None:
        self._loc = jnp.array(value, dtype=self.dtype)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value: ArrayLike) -> None:
        self._scale = jnp.array(value, dtype=self.dtype)

    def __str__(self) -> str:
        name = "Normal"
        loc = self.loc
        scale = self.scale
        return f"{name}({loc}, {scale})"

    def __repr__(self) -> str:
        name = self.__class__.__name__
        reprstr = f"{name}(loc={self.loc}, scale={self.scale},"
        reprstr += f" shape={self.shape}, dtype={self.dtype})"
        return reprstr

    def __copy__(self) -> "NormalPrior":
        return self.__class__(
            loc=self.loc, scale=self.scale, shape=self.shape, dtype=self.dtype
        )

    def sample(self, key: prng.PRNGKeyArray) -> Array:
        return normal_sample(
            key, loc=self.loc, scale=self.scale, shape=self.shape, dtype=self.dtype
        )

    def logpdf(self, x: ArrayLike) -> Array:
        return normal_logpdf(x=x, loc=self.loc, scale=self.scale)
