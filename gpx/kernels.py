import jax.numpy as jnp
from jax import vmap


# =============================================================================
# Kernel Mappings
# =============================================================================


def kmap(kernel, x1, x2, params):
    return vmap(lambda x: vmap(lambda y: kernel(x, y, params))(x2))(x1)


# =============================================================================
# Kernel Functions
# =============================================================================


def squared_exponential_kernel(x1, x2, params):
    z1 = x1 / params["lengthscale"]
    z2 = x2 / params["lengthscale"]
    d2 = jnp.sum((z1 - z2) ** 2)
    return jnp.exp(-d2)


def matern12_kernel(x1, x2, params):
    z1 = x1 / params["lengthscale"]
    z2 = x2 / params["lengthscale"]
    d = jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return jnp.exp(-d)


def matern32_kernel(x1, x2, params):
    z1 = x1 / params["lengthscale"]
    z2 = x2 / params["lengthscale"]
    d = jnp.sqrt(3.0) * jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return (1.0 + d) * jnp.exp(-d)


def matern52_kernel(x1, x2, params):
    z1 = x1 / params["lengthscale"]
    z2 = x2 / params["lengthscale"]
    d = jnp.sqrt(5.0) * jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return (1.0 + d + d**2 / 3.0) * jnp.exp(-d)


# =============================================================================
# Kernel abbreviations
# =============================================================================

se_kernel = squared_exponential_kernel
rbf_kernel = squared_exponential_kernel
m12_kernel = matern12_kernel
m32_kernel = matern32_kernel
m52_kernel = matern52_kernel
