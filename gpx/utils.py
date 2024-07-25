# GPX: gaussian process regression in JAX
# Copyright (C) 2023  GPX authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
