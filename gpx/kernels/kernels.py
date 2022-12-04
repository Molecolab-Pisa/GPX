import functools
import jax.numpy as jnp
from jax import vmap

from ..utils import squared_distances


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
# Kernel Functions
# =============================================================================

def squared_exponential_kernel(x1, x2, params):
    z1 = x1 / params['lengthscale']
    z2 = x2 / params['lengthscale']
    d2 = squared_distances(z1, z2)
    return jnp.exp(-d2)

#def squared_exponential_kernel_base(x1, x2, params):
#    z1 = x1 / params["lengthscale"]
#    z2 = x2 / params["lengthscale"]
#    d2 = jnp.sum((z1 - z2) ** 2)
#    return jnp.exp(-d2)
#
#squared_exponential_kernel = kernelize(squared_exponential_kernel_base)


def matern12_kernel_base(x1, x2, params):
    z1 = x1 / params["lengthscale"]
    z2 = x2 / params["lengthscale"]
    d = jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return jnp.exp(-d)

matern12_kernel = kernelize(matern12_kernel_base)


def matern32_kernel_base(x1, x2, params):
    z1 = x1 / params["lengthscale"]
    z2 = x2 / params["lengthscale"]
    d = jnp.sqrt(3.0) * jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return (1.0 + d) * jnp.exp(-d)

matern32_kernel = kernelize(matern32_kernel_base)


def matern52_kernel_base(x1, x2, params):
    z1 = x1 / params["lengthscale"]
    z2 = x2 / params["lengthscale"]
    d = jnp.sqrt(5.0) * jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return (1.0 + d + d**2 / 3.0) * jnp.exp(-d)

matern52_kernel = kernelize(matern52_kernel_base)


# =============================================================================
# Kernel abbreviations
# =============================================================================

se_kernel = squared_exponential_kernel
rbf_kernel = squared_exponential_kernel
m12_kernel = matern12_kernel
m32_kernel = matern32_kernel
m52_kernel = matern52_kernel


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'kernelize',
    'squared_exponential_kernel',
    'se_kernel',
    'rbf_kernel',
    'matern12_kernel',
    'm12_kernel',
    'matern32_kernel',
    'm32_kernel',
    'matern52_kernel',
    'm52_kernel',
]
