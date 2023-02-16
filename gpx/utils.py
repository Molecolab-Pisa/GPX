from typing import Any

import jax.numpy as jnp
from jax import jit

Array = Any


# =============================================================================
# Operations
# =============================================================================


@jit
def squared_distances(x1: Array, x2: Array) -> Array:
    jitter = 1e-12
    x1s = jnp.sum(jnp.square(x1), axis=-1)
    x2s = jnp.sum(jnp.square(x2), axis=-1)
    dist = x1s[:, jnp.newaxis] - 2 * jnp.dot(x1, x2.T) + x2s + jitter
    return dist


# =============================================================================
# Transformation Functions
# =============================================================================


@jit
def softplus(x: Array) -> Array:
    return jnp.logaddexp(x, 0.0)


@jit
def inverse_softplus(x: Array) -> Array:
    return jnp.log(jnp.expm1(x))


# =============================================================================
# Parameters Handling
# =============================================================================


# def split_params(params):
#    kernel_params = params["kernel_params"]
#    sigma = params["sigma"]
#    return kernel_params, sigma
#
#
# def transform_parameters(params, transform, ignore=None):
#    if ignore is not None:
#        tmp = params.copy()
#        tmp = {key: value for key, value in tmp.items() if key in ignore}
#        params = {key: value for key, value in params.items() if key not in ignore}
#        transformed = tree_map(lambda p: transform(p), params)
#        for key in tmp:
#            transformed[key] = tmp[key]
#    else:
#        transformed = tree_map(lambda p: transform(p), params)
#    return transformed
#
#
# def constrain_parameters(params, transform=softplus, ignore=None):
#    return transform_parameters(params, transform=transform, ignore=ignore)
#
#
# def unconstrain_parameters(params, transform=inverse_softplus, ignore=None):
#    return transform_parameters(params, transform=transform, ignore=ignore)


# def flatten_arrays(arrays):
#    shapes = [a.shape for a in arrays]
#    arrays = [a.reshape(-1) for a in arrays]
#    return arrays, shapes
#
#
# def unflatten_arrays(arrays, shapes):
#    if len(arrays) != len(shapes):
#        raise RuntimeError('Incompatible number of shapes/arrays')
#    return [a.reshape(s) for a, s in zip(arrays, shapes)]
