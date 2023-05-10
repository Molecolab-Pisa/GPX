import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpy.testing import assert_allclose
from scipy.special import gamma, kv

from gpx.kernels.kernels import (
    linear_kernel,
    m12_kernel,
    m32_kernel,
    m52_kernel,
    squared_exponential_kernel,
)
from gpx.parameters import Parameter
from gpx.priors import NormalPrior
from gpx.utils import inverse_softplus, softplus


# ============================================================================
# Reference kernels
# ============================================================================
def reference_linear_kernel(x1, x2, params):
    n1, m1 = x1.shape
    n2, _ = x2.shape
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            for l in range(m1):
                K[i, j] = K[i, j] + x1[i, l] * x2.T[l, j]
    return K


def reference_squared_exponential_kernel(x1, x2, params):
    n1, _ = x1.shape
    n2, _ = x2.shape
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist = x1[i] - x2[j]
            dist2 = jnp.dot(dist.T, dist)
            K[i, j] = jnp.exp(-dist2 / params["lengthscale"].value ** 2)
    return K


def reference_matern_kernel(x1, x2, nu, params):
    n1, _ = x1.shape
    n2, _ = x2.shape
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist = x1[i] - x2[j]
            dist = jnp.dot(dist.T, dist) ** 0.5
            dist = dist / params["lengthscale"].value
            fact = jnp.sqrt(2 * nu) * dist
            K[i, j] = ((2.0 ** (1.0 - nu)) / gamma(nu)) * (fact**nu) * kv(nu, fact)
    return K


# ============================================================================
# Linear kernel
# ============================================================================
@pytest.mark.parametrize("dim", [1, 10])
def test_linear_kernel(dim):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, dim))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, dim))
    params = {}

    K = linear_kernel(X1, X2, params)
    K_ref = reference_linear_kernel(X1, X2, params)

    assert_allclose(K, K_ref)


# ============================================================================
# Squared exponential kernel
# ============================================================================
@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_squared_exponential_kernel(dim, lengthscale):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, dim))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, dim))
    params = {
        "lengthscale": Parameter(
            lengthscale, True, softplus, inverse_softplus, NormalPrior()
        )
    }

    K = squared_exponential_kernel(X1, X2, params)
    K_ref = reference_squared_exponential_kernel(X1, X2, params)

    assert_allclose(K, K_ref)


# ============================================================================
# Matern kernel
# ============================================================================
@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_matern12_kernel(dim, lengthscale):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, dim))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, dim))
    params = {
        "lengthscale": Parameter(
            lengthscale, True, softplus, inverse_softplus, NormalPrior()
        )
    }

    K = m12_kernel(X1, X2, params)
    K_ref = reference_matern_kernel(X1, X2, 1.0 / 2, params)

    assert_allclose(K, K_ref)


@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_matern32_kernel(dim, lengthscale):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, dim))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, dim))
    params = {
        "lengthscale": Parameter(
            lengthscale, True, softplus, inverse_softplus, NormalPrior()
        )
    }

    K = m32_kernel(X1, X2, params)
    K_ref = reference_matern_kernel(X1, X2, 3.0 / 2, params)

    assert_allclose(K, K_ref)


@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_matern52_kernel(dim, lengthscale):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, dim))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, dim))
    params = {
        "lengthscale": Parameter(
            lengthscale, True, softplus, inverse_softplus, NormalPrior()
        )
    }

    K = m52_kernel(X1, X2, params)
    K_ref = reference_matern_kernel(X1, X2, 5.0 / 2, params)

    assert_allclose(K, K_ref)
