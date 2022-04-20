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
    return jnp.exp(-jnp.sum(z1 - z2) ** 2)


# =============================================================================
# Kernel abbreviations
# =============================================================================

# Squared Exponential Kernel
se_kernel = squared_exponential_kernel
rbf_kernel = squared_exponential_kernel
