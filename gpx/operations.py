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

from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax
from jax.tree_util import tree_map
from jax.typing import ArrayLike


@jit
def recover_first_axis(xs: Any) -> Any:
    "adds a singleton axis at position 0 to each x in xs"
    return tree_map(lambda x: jnp.expand_dims(x, axis=0), xs)


@jit
def update_row_diagonal(row: ArrayLike, index: ArrayLike, value: ArrayLike) -> Array:
    """update the diagonal of a 'row'

    The kernel K in the kernel-vector product K∙x is occasionally jitted on
    the diagonal, either with a small jitter value and/or with the likelihood
    noise σ². This function adds the jitter along the diagonal, given the
    `index`-th row of the kernel K.
    Note that sometimes a row is not strictly a row (i.e., shape (1, n)) but
    a stripe (i.e., shape (nd, n)), for example if the kernel K is a hessian
    kernel, where the first dimension equals the number of derivative values
    for each point. This function is built to deal with that option too.
    """
    num_rows = row.shape[0]
    start_indices = (0, num_rows * index)
    add_to_diag = value * jnp.eye(num_rows)
    size = (num_rows, num_rows)
    new_diag = lax.dynamic_slice(row, start_indices, size) + add_to_diag
    row = lax.dynamic_update_slice(row, new_diag, start_indices)
    return row


def rowfun_to_matvec(
    row_fun: Callable[ArrayLike, Array],
    init_val: Tuple[ArrayLike],
    update_diag: Optional[bool] = False,
    diag_value: Optional[ArrayLike] = None,
) -> Callable[ArrayLike, Array]:
    """transform a function computing a row of A in A∙x into a matvec(z)

    This function transforms a function `row_fun`, computing a single
    row of the matrix A in the product A∙x, into a matvec function
    that computes the product A∙x without ever instantiating A.

    `row_fun` can accept any number of arguments which are looped onto.
    For example, it could be:

    >>> def row_fun(x1s):
    ...     return kernel(x1s, x, kernel_params)

    which computes a single row of a certain kernel matrix, with `x1s`
    being a single point in `x`. In order to do so, the `init_val` should
    be given as `(x,)` (note the comma). To compute the hessian kernel instead:

    >>> def row_fun(x1s, j1s):
    ...     return kernel.d01kj(x1s, x, kernel_params, j1s, jacobian)

    and `init_val` should be given as `(x, jacobian)`.

    It is possible to add some quantity to the diagonal of A with
    `update_diag=True` and by providing the value inside `diag_value`.
    Only a single scalar value is currently supported.

    Args:
        row_fun: function computing a single row of A
        init_val: variables used to execute row_fun
        update_diag: whether a scalar should be added to the diagonal of A
        diag_value: value to be added to the diagonal of A
    Returns:
        matvec: function computing A∙x
    """
    if update_diag:

        @jit
        def matvec(z: ArrayLike) -> Array:
            def update_row(carry, xs):
                xs = recover_first_axis(xs)
                row = row_fun(*xs)
                row = update_row_diagonal(row, carry, diag_value)
                rowvec = jnp.dot(row, z)
                carry = carry + 1
                return carry, rowvec

            _, res = lax.scan(update_row, 0, init_val)
            res = jnp.concatenate(res, axis=0)
            return res

    else:

        @jit
        def matvec(z: ArrayLike) -> Array:
            def update_row(carry, xs):
                xs = recover_first_axis(xs)
                row = row_fun(*xs)
                rowvec = jnp.dot(row, z)
                return carry, rowvec

            _, res = lax.scan(update_row, 0, init_val)
            res = jnp.concatenate(res, axis=0)
            return res

    return matvec
