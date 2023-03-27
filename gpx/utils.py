from __future__ import annotations

import jax.numpy as jnp
from jax import jit

# =============================================================================
# Operations
# =============================================================================


@jit
def squared_distances(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    jitter = 1e-12
    x1s = jnp.sum(jnp.square(x1), axis=-1)
    x2s = jnp.sum(jnp.square(x2), axis=-1)
    dist = x1s[:, jnp.newaxis] - 2 * jnp.dot(x1, x2.T) + x2s + jitter
    return dist


# =============================================================================
# Transformation Functions
# =============================================================================


@jit
def softplus(x: jnp.ndarray) -> jnp.ndarray:
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
def inverse_softplus(x: jnp.ndarray) -> jnp.ndarray:
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
def identity(x: jnp.ndarray) -> jnp.ndarray:
    """identity transformation

    Computes the identity transformation of the input x,
    leaving it intact.
    """
    return x
