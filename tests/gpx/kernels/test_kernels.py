
import pytest
import numpy as np
from numpy.testing import assert_allclose

import jax
import jax.numpy as jnp
from jax import random

import gpx
from gpx.kernels import squared_exponential_kernel


# We want float64 enabled in JAX when importing gpx
def test_float64():
    assert jax.config.x64_enabled == True


# ============================================================================
# Reference kernels
# ============================================================================
def reference_squared_exponential_kernel(x1, x2, params):
    n1, _ = x1.shape
    n2, _ = x2.shape
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist = x1[i] - x2[j]
            dist2 = jnp.dot(dist.T, dist)
            K[i, j] = jnp.exp(-dist2 / params['lengthscale']**2)
    return K


# ============================================================================
# Squared exponential kernel
# ============================================================================
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_squared_exponential_kernel(lengthscale):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, 10))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, 10))
    params = {'lengthscale': lengthscale}

    K = squared_exponential_kernel(X1, X2, params)
    K_ref = reference_squared_exponential_kernel(X1, X2, params)

    assert_allclose(K, K_ref)


# test_matern52_kernel
# test_matern32_kernel
# test_matern12_kernel