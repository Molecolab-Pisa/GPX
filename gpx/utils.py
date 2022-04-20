import jax.numpy as jnp


# =============================================================================
# Transformation Functions
# =============================================================================


def softplus(x):
    return jnp.logaddexp(x, 0.0)


# =============================================================================
# Parameters Handling
# =============================================================================


def split_params(params):
    kernel_params = params["kernel_params"]
    sigma = params["sigma"]
    return kernel_params, sigma
