from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax
from jax.tree_util import tree_map
from jax.typing import ArrayLike


@jit
def recover_first_axis(xs):
    "adds a singleton axis at position 0 to each x in xs"
    return tree_map(lambda x: jnp.expand_dims(x, axis=0), xs)


@jit
def update_row_diagonal(row: ArrayLike, index: ArrayLike, value: ArrayLike):
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
