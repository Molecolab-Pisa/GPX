from __future__ import annotations

import functools
from functools import partial
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, vmap

# def _warn_experimental(funcname):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             warnings.warn(
#                 f"{funcname} is still experimental and not tested.", stacklevel=2
#             )
#             return func(*args, **kwargs)
#
#         return wrapper
#
#     return decorator


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
        def kernel(x1, x2, params, active_dims=None):
            _kernel_func = lambda x1: vmap(  # noqa: E731
                lambda x2: kernel_func(x1, x2, params, active_dims=active_dims)
            )

            @jax.checkpoint
            def update_row(carry, x1s):
                gram_row = _kernel_func(x1s)(x2)
                return carry, gram_row

            _, gram = jax.lax.scan(update_row, 0, x1)
            return gram

    else:

        @functools.wraps(kernel_func)
        @jit
        def kernel(x1, x2, params, active_dims=None):
            return vmap(
                lambda x: vmap(
                    lambda y: kernel_func(x, y, params, active_dims=active_dims)
                )(x2)
            )(x1)

    return kernel


# =============================================================================
# Derivative Kernel Decorator: low level decorators
# =============================================================================


def _grad0_kernelize(kernel_func: Callable) -> Callable:
    kernel_func = jacrev(kernel_func, argnums=0)

    @functools.wraps(kernel_func)
    @jit
    def kernel(x1, x2, params, active_dims=None):
        n, nf = x1.shape
        m, _ = x2.shape

        _kernel_func = lambda x1: vmap(  # noqa: E731
            lambda x2: kernel_func(x1, x2, params, active_dims=active_dims)
        )

        @jax.checkpoint
        def update_row(carry, x1s):
            nabla_k = jnp.atleast_2d(_kernel_func(x1s)(x2)).T
            return carry, nabla_k

        _, gram = jax.lax.scan(update_row, 0, x1)
        return gram.reshape((n * nf, m))

    return kernel


def _grad0jac_kernelize(kernel_func: Callable) -> Callable:
    kernel_func = jacrev(kernel_func, argnums=0)

    @functools.wraps(kernel_func)
    @jit
    def kernel(x1, x2, params, jacobian, active_dims=None):
        n, _ = x1.shape
        m, _ = x2.shape
        _, _, jv = jacobian.shape

        gram = jnp.zeros((n * jv, m))

        _kernel_func = lambda x1: vmap(  # noqa: E731
            lambda x2: kernel_func(x1, x2, params, active_dims=active_dims)
        )

        @jax.checkpoint
        def update_row(i, gram):
            nabla_k = _kernel_func(x1[i])(x2)
            # k = m, i = nf, j = jv
            # note: active_dims ensures that nabla_k is zero if a feature
            # is not included, so there's no need to also use active_dims
            # on the jacobian
            nabla_k = jnp.einsum("ki,ij->jk", nabla_k, jacobian[i])
            return jax.lax.dynamic_update_slice(
                gram, update=nabla_k, start_indices=(i * jv, 0)
            )

        return jax.lax.fori_loop(0, n, update_row, gram)

    return kernel


def _grad1_kernelize(kernel_func: Callable) -> Callable:
    kernel_func = jacrev(kernel_func, argnums=1)

    @functools.wraps(kernel_func)
    @jit
    def kernel(x1, x2, params, active_dims=None):
        n, _ = x1.shape
        m, mf = x2.shape

        _kernel_func = lambda x1: vmap(  # noqa: E731
            lambda x2: kernel_func(x1, x2, params, active_dims=active_dims)
        )

        @jax.checkpoint
        def update_row(carry, x1s):
            nabla_k = jnp.atleast_2d(_kernel_func(x1s)(x2))
            return carry, nabla_k

        _, gram = jax.lax.scan(update_row, 0, x1)
        return gram.reshape((n, m * mf))

    return kernel


def _grad1jac_kernelize(kernel_func: Callable) -> Callable:
    kernel_func = jacrev(kernel_func, argnums=1)

    @functools.wraps(kernel_func)
    @jit
    def kernel(x1, x2, params, jacobian, active_dims=None):
        n, _ = x1.shape
        m, _ = x2.shape
        _, _, jv = jacobian.shape

        gram = jnp.zeros((n, m * jv))

        _kernel_func = lambda x2: vmap(  # noqa: E731
            lambda x1: kernel_func(x1, x2, params, active_dims=active_dims)
        )

        @jax.checkpoint
        def update_col(j, gram):
            nabla_k = _kernel_func(x2[j])(x1)
            # k = m, i = mf, j = jv
            # note: active_dims ensures that nabla_k is zero if a feature
            # is not included, so there's no need to also use active_dims
            # on the jacobian
            nabla_k = jnp.einsum("ki,ij->kj", nabla_k, jacobian[j])
            return jax.lax.dynamic_update_slice(
                gram, update=nabla_k, start_indices=(0, j * jv)
            )

        return jax.lax.fori_loop(0, m, update_col, gram)

    return kernel


def _grad01_kernelize(kernel_func: Callable) -> Callable:
    kernel_func = jacfwd(jacrev(kernel_func, argnums=0), argnums=1)

    @functools.wraps(kernel_func)
    @jit
    def kernel(x1, x2, params, active_dims=None):
        n, nf = x1.shape
        m, mf = x2.shape

        _kernel_func = lambda x1: vmap(  # noqa: E731
            lambda x2: kernel_func(x1, x2, params, active_dims=active_dims), out_axes=1
        )

        @jax.checkpoint
        def update_row(carry, x1s):
            hess_k = jnp.atleast_2d(_kernel_func(x1s)(x2))
            return carry, hess_k

        _, gram = jax.lax.scan(update_row, 0, x1)
        return gram.reshape((n * nf, m * mf))

    return kernel


def _grad01jac_kernelize(kernel_func: Callable) -> Callable:
    kernel_func = jacfwd(jacrev(kernel_func, argnums=0), argnums=1)

    @functools.wraps(kernel_func)
    @jit
    def kernel(x1, x2, params, jacobian1, jacobian2, active_dims=None):
        n, _ = x1.shape
        m, _ = x2.shape
        _, _, j1v = jacobian1.shape
        _, _, j2v = jacobian2.shape

        gram = jnp.zeros((n * j1v, m * j2v))

        @jax.checkpoint
        def update_row(i, gram):
            def update_col(j, gram):
                nabla_k = kernel_func(x1[i], x2[j], params, active_dims=active_dims)
                # note: active_dims ensures that nabla_k is zero if a feature
                # is not included, so there's no need to also use active_dims
                # on the jacobian
                nabla_k = jnp.einsum(
                    "ai,ab,bj->ij", jacobian1[i], nabla_k, jacobian2[j]
                )
                return jax.lax.dynamic_update_slice(
                    gram, update=nabla_k, start_indices=(i * j1v, j * j2v)
                )

            return jax.lax.fori_loop(0, m, update_col, gram)

        return jax.lax.fori_loop(0, n, update_row, gram)

    return kernel


# =============================================================================
# Derivative Kernel Decorator: high level decorators
# =============================================================================


def grad0_kernelize(kernel_func: Callable, with_jacob: bool = False):
    if with_jacob:
        return _grad0jac_kernelize(kernel_func)
    else:
        return _grad0_kernelize(kernel_func)


def grad1_kernelize(kernel_func: Callable, with_jacob: bool = False):
    if with_jacob:
        return _grad1jac_kernelize(kernel_func)
    else:
        return _grad1_kernelize(kernel_func)


def grad01_kernelize(kernel_func: Callable, with_jacob: bool = False):
    if with_jacob:
        return _grad01jac_kernelize(kernel_func)
    else:
        return _grad01_kernelize(kernel_func)


def grad_kernelize(
    argnums: Union[int, Tuple[int, int]], with_jacob: bool = False
) -> Callable:
    """Kernelizes the input kernel with respect to the dimension
    specified in argnums.

    Only argnums == 0, 1, (0, 1) is available.
    """
    if argnums == 0:
        return partial(grad0_kernelize, with_jacob=with_jacob)
    elif argnums == 1:
        return partial(grad1_kernelize, with_jacob=with_jacob)
    elif argnums == (0, 1):
        return partial(grad01_kernelize, with_jacob=with_jacob)
    else:
        raise ValueError(
            f"argnums={argnums} is not valid. Allowed argnums: 0, 1, (0, 1)"
        )
