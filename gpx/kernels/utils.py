from typing import Any, Callable, Tuple, Union

import functools
from functools import partial
import jax.numpy as jnp
from jax import vmap, jvp, jacrev, jacfwd

Array = Any


# =============================================================================
# Kernel Decorator
# =============================================================================


def kernelize(kernel_func: Callable) -> Callable:
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

    Returns:
        A vectorized kernel function that applies the original `kernel_func`
        to batches of data.
    """

    @functools.wraps(kernel_func)
    def kernel(x1, x2, params):
        return vmap(lambda x: vmap(lambda y: kernel_func(x, y, params))(x2))(x1)

    return kernel


# =============================================================================
# Derivative kernel decorators
# =============================================================================


def _grad0_kernelize(k: Callable) -> Callable:
    return kernelize(jacrev(k, argnums=0))


def _grad1_kernelize(k: Callable) -> Callable:
    return kernelize(jacrev(k, argnums=1))


def _grad01_kernelize(k: Callable) -> Callable:
    return kernelize(jacfwd(jacrev(k, argnums=0), argnums=1))


def grad0_kernelize(k: Callable) -> Callable:
    """Kernelizes the kernel k and makes a derivative kernel

    d/d0(k) = cov(w, y) with y = f(x) and w = d/dx(f)(x).

    Returns:
    d0_kernel: derivative kernel with respect to the first argument
    """
    d0k = _grad0_kernelize(k)

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


def grad1_kernelize(k: Callable) -> Callable:
    """Kernelizes the kernel k and makes a derivative kernel

    d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x).

    Returns:
    d1_kernel: derivative kernel with respect to the second argument
    """
    d1k = _grad1_kernelize(k)

    def wrapper(x1, x2, params):
        """Derivative kernel with respect to the second argument.

        d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x).
        """
        # n, m, d = x1.shape[0], x2.shape[0], x1.shape[1]
        gram = d1k(x1, x2, params)
        return gram
        # return jnp.reshape(gram, (n, m * d))

    return wrapper


def grad01_kernelize(k: Callable) -> Callable:
    """Kernelizes the kernel k and makes a derivative kernel

    d^2/d0d1(k) = cov(w, w) with y = f(x) and w = d/dx(f)(x).

    Returns:
    d0d1_kernel: derivative kernel with respect to the first and
                 second argument
    """
    d01k = _grad01_kernelize(k)

    def wrapper(x1, x2, params):
        # n, m, d = x1.shape[0], x2.shape[0], x1.shape[1]
        gram = d01k(x1, x2, params)
        return jnp.transpose(gram, axes=(2, 0, 1, 3))
        # gram = jnp.transpose(gram, axes=(0, 2, 1, 3))
        # return jnp.reshape(gram, (n * d, m * d))

    return wrapper


def grad_kernelize(argnums: Union[int, Tuple[int, int]]) -> Callable:
    """Kernelizes the input kernel with respect to the dimension
    specified in argnums.

    Only argnums == 0, 1, (0, 1) is available.
    """
    if argnums == 0:
        return grad0_kernelize
    elif argnums == 1:
        return grad1_kernelize
    elif argnums == (0, 1):
        return grad01_kernelize
    else:
        raise ValueError(
            f"argnums={argnums} is not valid. Allowed argnums: 0, 1, (0, 1)"
        )


# =============================================================================
# Derivative kernel functions
# =============================================================================
# WARNING: these functions do not work in more than one dimension.
#          in that case use the kernelize decorators above.


def d0_k(k: Callable) -> Tuple[Callable, Callable]:
    docstr = """
    Derivative kernel with respect to the first argument.

        d/d0(k) = cov(w, y) with y = f(x) and w = d/dx(f)(x).

    Returns:
    kernel: original kernel
    d0_kernel: derivative kernel with respect to the first argument
    """

    def wrapper(x1, x2, params):
        return jvp(partial(k, x2=x2, params=params), (x1,), (jnp.ones(x1.shape),))

    wrapper.__doc__ = docstr + k.__doc__ if k.__doc__ is not None else docstr

    return wrapper


def d1_k(k: Callable) -> Tuple[Callable, Callable]:
    docstr = """
    Derivative kernel with respect to the seoncd argument.

        d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x).

    Returns:
    kernel: original kernel
    d1_kernel: derivative kernel with respect to the second argument
    """

    def wrapper(x1, x2, params):
        return jvp(partial(k, x1, params=params), (x2,), (jnp.ones(x2.shape),))

    wrapper.__doc__ = docstr + k.__doc__ if k.__doc__ is not None else docstr

    return wrapper


def d0d1_k(k: Callable) -> Tuple[Callable, Callable, Callable, Callable]:
    docstr = """
    Derivative kernel with respect to the second argument.

        d^2/d0d1(k) = cov(w, w) with y = f(x) and w = d/dx(f)(x).

    Returns:
    kernel: original kernel
    d0_kernel: derivative kernel with respect to the first argument
    d1_kernel: derivative kernel with respect to the second argument
    d0d1_kernel: derivative kernel with respect to the first and second argument
    """

    def wrapper(x1, x2, params):
        return jvp(partial(d0_k(k), x1, params=params), (x2,), (jnp.ones(x2.shape),))

    wrapper.__doc__ = docstr + k.__doc__ if k.__doc__ is not None else docstr

    return wrapper


def grad_kernel(k: Callable, argnums: Union[int, Tuple[int, int]]):
    """
    Generate a derivative kernel function.

    argnums == 0:      d/d0(k) = cov(w, y) with y = f(x) and w = d/dx(f)(x)
    argnums == 1:      d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x)
    argnums == (0, 1): d^2/d0d1(k) = cov(w, w) with y = f(x) and w = d/dx(f)(x)

    Returns:
    func with arguments (x1, x2, params)
    """
    if argnums == 0:
        return d0_k(k)
    elif argnums == 1:
        return d1_k(k)
    elif argnums == (0, 1):
        return d0d1_k(k)
    else:
        raise ValueError(
            f"argnums={argnums} is not valid. Allowed argnums: 0, 1, (0, 1)"
        )
