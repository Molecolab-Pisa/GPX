from __future__ import annotations
from typing import Callable, Tuple, Union

import functools
from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap, jit, jacrev, jacfwd


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


# =============================================================================
# Derivative Kernel Decorator
# =============================================================================
# TODO: right now we have a `vmap` version of the kernelizer, and a `lax`
#       version. The lax one is ok for big datasets. Still, it stores in memory
#       the full derivative kernel before taking the product with the jacobian
#       this is, in general, memory inefficient. A better way to implement this
#       is to perform the matrix jacobian product directly inside the for loop
#       of jax.lax. __implement this__.
#       Maybe is ok to have a _grad0_kernelize function that yields the derivative
#       kernel (in which case, the current shape is wrong and has to be changed)
#       and a _grad0jacob_kernelize function that yields the derivative kernel
#       multiplied by the jacobian, and leave to grad0_kernelize the duty to
#       dispatch the kernelization according to a `with_jacob` boolean argument.


def _grad0_kernelize(kernel_func: Callable, lax: bool = False) -> Callable:
    kernel_func = jacrev(kernel_func, argnums=0)

    if lax:

        @functools.wraps(kernel_func)
        @jit
        def kernel(x1, x2, params):
            n, nf = x1.shape
            m, _ = x2.shape
            gram = jnp.zeros((n, m, nf))

            def update_row(i, gram):
                def update_col(j, gram):
                    return gram.at[i, j, :].set(kernel_func(x1[i], x2[j], params))

                return jax.lax.fori_loop(0, m, update_col, gram)

            return jax.lax.fori_loop(0, n, update_row, gram)

    else:

        # vmap version is compatible with jacrev
        kernel = kernelize(kernel_func, lax=False)

    return kernel


def _grad1_kernelize(kernel_func: Callable, lax: bool = False) -> Callable:
    kernel_func = jacrev(kernel_func, argnums=1)

    if lax:

        @functools.wraps(kernel_func)
        @jit
        def kernel(x1, x2, params):
            n, _ = x1.shape
            m, mf = x2.shape
            gram = jnp.zeros((n, m, mf))

            def update_row(i, gram):
                def update_col(j, gram):
                    return gram.at[i, j, :].set(kernel_func(x1[i], x2[j], params))

                return jax.lax.fori_loop(0, m, update_col, gram)

            return jax.lax.fori_loop(0, n, update_row, gram)

    else:

        # vmap version is compatible with jacrev
        kernel = kernelize(kernel_func, lax=False)

    return kernel


def _grad01_kernelize(kernel_func: Callable, lax: bool = True) -> Callable:
    kernel_func = jacfwd(jacrev(kernel_func, argnums=0), argnums=1)

    if lax:

        @functools.wraps(kernel_func)
        @jit
        def kernel(x1, x2, params):
            n, nf = x1.shape
            m, mf = x2.shape
            gram = jnp.zeros((n, m, nf, mf))

            def update_row(i, gram):
                def update_col(j, gram):
                    return gram.at[i, j, :, :].set(kernel_func(x1[i], x2[j], params))

                return jax.lax.fori_loop(0, m, update_col, gram)

            return jax.lax.fori_loop(0, n, update_row, gram)

    else:

        # vmap version is compatile with jacrev, jacfwd
        kernel = kernelize(kernel_func, lax=False)

    return kernel


def grad0_kernelize(k: Callable, lax: bool = True) -> Callable:
    """Kernelizes the kernel k and makes a derivative kernel

    d/d0(k) = cov(w, y) with y = f(x) and w = d/dx(f)(x).

    Returns:
    d0_kernel: derivative kernel with respect to the first argument
    """
    d0k = _grad0_kernelize(k, lax=lax)

    def wrapper(x1, x2, params, jacobian):
        """Derivative kernel with respect to the first argument.

        d/d0(k) = cov(w, y) with y = f(x) and w = d/dx(f)(x).
        """
        # n, m, d = x1.shape[0], x2.shape[0], x1.shape[1]
        gram = d0k(x1, x2, params)
        gram = jnp.transpose(gram, axes=(2, 1, 0))
        gram = jnp.einsum("ijk,ijl->jkl", jacobian, gram)
        n, d, m = gram.shape
        return jnp.reshape(gram, (n * d, m))
        # return jnp.transpose(gram, axes=(2, 1, 0))
        # gram = jnp.transpose(gram, axes=(0, 2, 1))
        # return jnp.reshape(gram, (n * d, m))

    wrapper.__doc__ = grad0_kernelize.__doc__

    return wrapper


def grad1_kernelize(k: Callable, lax: bool = True) -> Callable:
    """Kernelizes the kernel k and makes a derivative kernel

    d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x).

    Returns:
    d1_kernel: derivative kernel with respect to the second argument
    """
    d1k = _grad1_kernelize(k, lax=lax)

    def wrapper(x1, x2, params, jacobian):
        """Derivative kernel with respect to the second argument.

        d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x).
        """
        # n, m, d = x1.shape[0], x2.shape[0], x1.shape[1]
        gram = d1k(x1, x2, params)
        gram = jnp.einsum("ijk,kjl->ijl", gram, jacobian)
        m, n, d = gram.shape
        return jnp.reshape(gram, (m, n * d))
        # return jnp.reshape(gram, (n, m * d))

    return wrapper


def grad01_kernelize(k: Callable, lax: bool = True) -> Callable:
    """Kernelizes the kernel k and makes a derivative kernel

    d^2/d0d1(k) = cov(w, w) with y = f(x) and w = d/dx(f)(x).

    Returns:
    d0d1_kernel: derivative kernel with respect to the first and
                 second argument
    """
    d01k = _grad01_kernelize(k, lax=lax)

    def wrapper(x1, x2, params, jacobian):
        # n, m, d = x1.shape[0], x2.shape[0], x1.shape[1]
        gram = d01k(x1, x2, params)
        gram = jnp.transpose(gram, axes=(2, 0, 1, 3))
        gram = jnp.einsum("ijk,ijlm,mln->jkln", jacobian, gram, jacobian)
        d, n, m, e = gram.shape
        return jnp.reshape(gram, (n * d, m * e))
        # return jnp.transpose(gram, axes=(2, 0, 1, 3))
        # gram = jnp.transpose(gram, axes=(0, 2, 1, 3))
        # return jnp.reshape(gram, (n * d, m * d))

    return wrapper


def grad_kernelize(argnums: Union[int, Tuple[int, int]], lax: bool = True) -> Callable:
    """Kernelizes the input kernel with respect to the dimension
    specified in argnums.

    Only argnums == 0, 1, (0, 1) is available.
    """
    if argnums == 0:
        return partial(grad0_kernelize, lax=lax)
    elif argnums == 1:
        return partial(grad1_kernelize, lax=lax)
    elif argnums == (0, 1):
        return partial(grad01_kernelize, lax=lax)
    else:
        raise ValueError(
            f"argnums={argnums} is not valid. Allowed argnums: 0, 1, (0, 1)"
        )
