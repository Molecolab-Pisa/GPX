from __future__ import annotations

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

# =============================================================================
# Operations
# =============================================================================


@jit
def euclidean_distance(x1: ArrayLike, x2: ArrayLike) -> Array:
    """euclidean distance

    Euclidean distance between two points, each of shape (1, n_feats).
    This function uses the "double where trick" to ensure it is differentiable
    (yielding zeros) in edge cases, such as when the derivative involves
    a 0 / 0 operation arising from computing a distance between two identical
    points.
    """
    # l1 norm (differentiable)
    d1 = jnp.sum(jnp.abs(x1 - x2))
    d2 = jnp.sum((x1 - x2) ** 2)
    zeros = jnp.equal(d2, 0.0)
    # here we use the "double where" trick
    # if the distance is zero, substitute with 1.0
    # 1.0 are not used, but avoid propagating NaN values
    # in unused branch of autodiff
    # for more info, see:
    #   https://github.com/google/jax/issues/1052
    # and links therein
    d2 = jnp.where(zeros, jnp.ones_like(d2), d2)
    # return the differentiable l1 norm if the distance
    # is 0.0. This ensures the function is differentiable
    # in edge cases
    return jnp.where(zeros, d1, jnp.sqrt(d2))


@jit
def squared_distances(x1: ArrayLike, x2: ArrayLike) -> Array:
    """squared euclidean distances

    This is a memory-efficient implementation of the calculation of
    squared euclidean distances. Euclidean distances between `x1`
    of shape (n_samples1, n_feats) and `x2` of shape (n_samples2, n_feats)
    is evaluated by using the "euclidean distances trick":

        dist = X1 @ X1.T - 2 X1 @ X2.T + X2 @ X2.T

    Note: this function evaluates distances between batches of points
    """
    jitter = 1e-12
    x1s = jnp.sum(jnp.square(x1), axis=-1)
    x2s = jnp.sum(jnp.square(x2), axis=-1)
    dist = x1s[:, jnp.newaxis] - 2 * jnp.dot(x1, x2.T) + x2s + jitter
    return dist


# =============================================================================
# Transformation Functions
# =============================================================================


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
def identity(x: ArrayLike) -> Array:
    """identity transformation

    Computes the identity transformation of the input x,
    leaving it intact.
    """
    return x
