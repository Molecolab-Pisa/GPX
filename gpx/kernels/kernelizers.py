from __future__ import annotations
from typing import Callable

import functools
import jax
import jax.numpy as jnp
from jax import vmap, jit


# =============================================================================
# Kernel Decorator
# =============================================================================


def kernelize(kernel_func: Callable, lax: bool = True) -> Callable:
    """Decorator to promote a kernel function operating on single samples to a
       function operating on batches.

    With this decorator, you can write a function operating on a pair of
    samples, and vectorize it so that it accepts two batches of samples.
    Note that this may not be the fastest way to write your kernel.
    Still, it can be useful in the general setting, and to test the values
    of your kernel.

    Args:
        kernel_func: a function accepting three arguments: x1, x2, and params.
          x1 and x2 are two samples of data, while params is a dictionary of
          kernel parameters.
        lax: whether to use a kernelizer implemented with jax.lax operations
             (True) or a kernelizer implemented with vmap (False).
             The vmap version should be faster for small number
             of samples, but can break as the number of samples increases.
             The lax version is slower, but scales much better with the number
             of samples.

    Returns:
        A vectorized kernel function that applies the original `kernel_func`
        to batches of data.
    """
    if lax:

        @functools.wraps(kernel_func)
        @jit
        def kernel(x1, x2, params):
            n, _ = x1.shape
            m, _ = x2.shape
            gram = jnp.zeros((n, m))

            def update_row(i, gram):
                def update_col(j, gram):
                    return gram.at[i, j].set(kernel_func(x1[i], x2[j], params))

                return jax.lax.fori_loop(0, m, update_col, gram)

            return jax.lax.fori_loop(0, n, update_row, gram)

    else:

        @functools.wraps(kernel_func)
        @jit
        def kernel(x1, x2, params):
            return vmap(lambda x: vmap(lambda y: kernel_func(x, y, params))(x2))(x1)

    return kernel
