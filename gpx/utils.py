import jax.numpy as jnp


# =============================================================================
# Operations
# =============================================================================


def squared_distances(x1, x2):
    jitter = 1e-12
    x1s = jnp.sum(jnp.square(x1), axis=-1)
    x2s = jnp.sum(jnp.square(x2), axis=-1)
    dist = x1s[:, jnp.newaxis] - 2 * jnp.dot(x1, x2.T) + x2s
    return jnp.clip(dist, jitter)


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
