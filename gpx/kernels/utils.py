import functools
import jax.numpy as jnp
from jax import vmap, jit, jvp


# =============================================================================
# Kernel Decorator
# =============================================================================


def kernelize(kernel_func):
    """Decorator to promote a kernel function operating on single samples to a
       function operating on batches.

    With this decorator, you can write a function operating on a pair of samples,
    and vectorize it so that it accepts two batches of samples.
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
# Derivative Kernels
# =============================================================================


def d0_k(k):
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


def d1_k(k):
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


def d0d1_k(k):
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


def grad_kernel(k, argnums):
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



# =============================================================================
# Export
# =============================================================================

__all__ = [
    'd0_k',
    'd1_k',
    'd0d1_k',
    'grad_kernel',
    'kernelize',
]
