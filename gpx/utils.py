import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_map


# =============================================================================
# Operations
# =============================================================================


@jit
def squared_distances(x1, x2):
    jitter = 1e-12
    x1s = jnp.sum(jnp.square(x1), axis=-1)
    x2s = jnp.sum(jnp.square(x2), axis=-1)
    dist = x1s[:, jnp.newaxis] - 2 * jnp.dot(x1, x2.T) + x2s
    return jnp.clip(dist, jitter)


# =============================================================================
# Transformation Functions
# =============================================================================


@jit
def softplus(x):
    return jnp.logaddexp(x, 0.0)


@jit
def inverse_softplus(x):
    return jnp.log(jnp.expm1(x))


# =============================================================================
# Parameters Handling
# =============================================================================


def split_params(params):
    kernel_params = params["kernel_params"]
    sigma = params["sigma"]
    return kernel_params, sigma


def constrain_parameters(params, transform=softplus):
    return tree_map(lambda p: transform(p), params)


def uncostrain_parameters(params, transform=inverse_softplus):
    return tree_map(lambda p: transform(p), params)
