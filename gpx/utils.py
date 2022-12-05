from tabulate import tabulate

import numpy as np

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


def transform_parameters(params, transform, ignore=None):
    if ignore is not None:
        tmp = params.copy()
        tmp = {key: value for key, value in tmp.items() if key in ignore}
        params = {key: value for key, value in params.items() if key not in ignore}
        transformed = tree_map(lambda p: transform(p), params)
        for key in tmp:
            transformed[key] = tmp[key]
    else:
        transformed = tree_map(lambda p: transform(p), params)
    return transformed


def constrain_parameters(params, transform=softplus, ignore=None):
    return transform_parameters(params, transform=transform, ignore=ignore)


def uncostrain_parameters(params, transform=inverse_softplus, ignore=None):
    return transform_parameters(params, transform=transform, ignore=ignore)


def flatten_arrays(arrays):
    shapes = [a.shape for a in arrays]
    arrays = [a.reshape(-1) for a in arrays]
    return arrays, shapes


def unflatten_arrays(arrays, shapes):
    if len(arrays) != len(shapes):
        raise RuntimeError('Incompatible number of shapes/arrays')
    return [a.reshape(s) for a, s in zip(arrays, shapes)]


# =============================================================================
# Printing
# =============================================================================


def print_model(model, tablefmt='simple_grid'):
    params = model.params.copy()
    kernel_params = params.pop('kernel_params')

    headers = ['name', 'type', 'dtype', 'shape', 'value']
    string_repr = lambda v: np.array2string(v, edgeitems=1, threshold=1)
    get_info = lambda v: (type(v), v.dtype, v.shape, string_repr(v))

    fields = [['kernel '+k]+list(get_info(v)) for k, v in kernel_params.items()]
    fields += [[k]+list(get_info(v)) for k, v in params.items()]
    #fields += [['sigma']+list(get_info(sigma))]

    with np.printoptions(edgeitems=0):
        print(tabulate(fields, headers=headers, tablefmt=tablefmt))
