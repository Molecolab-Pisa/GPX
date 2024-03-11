import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import gamma, kv

from gpx.bijectors import Softplus
from gpx.kernels import (
    Constant,
    ExpSinSquared,
    Linear,
    Matern12,
    Matern32,
    Matern52,
    Polynomial,
    Prod,
    SquaredExponential,
    Sum,
    constant_kernel_base,
    expsinsquared_kernel_base,
    linear_kernel_base,
    matern12_kernel_base,
    matern32_kernel_base,
    matern52_kernel_base,
    no_intercept_polynomial_kernel_base,
    polynomial_kernel_base,
    squared_exponential_kernel_base,
)
from gpx.kernels.kernels import (
    Kernel,
    linear_kernel,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    squared_exponential_kernel,
)
from gpx.kernels.operations import prod_kernels, sum_kernels
from gpx.parameters import Parameter
from gpx.priors import NormalPrior


# ============================================================================
# Reference kernels
# ============================================================================
def reference_linear_kernel(x1, x2, params):
    n1, m1 = x1.shape
    n2, _ = x2.shape
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            for k in range(m1):
                K[i, j] = K[i, j] + x1[i, k] * x2.T[k, j]
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
    params = {"lengthscale": Parameter(lengthscale, True, Softplus(), NormalPrior())}

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
    params = {"lengthscale": Parameter(lengthscale, True, Softplus(), NormalPrior())}

    K = matern12_kernel(X1, X2, params)
    K_ref = reference_matern_kernel(X1, X2, 1.0 / 2, params)

    assert_allclose(K, K_ref)


@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_matern32_kernel(dim, lengthscale):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, dim))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, dim))
    params = {"lengthscale": Parameter(lengthscale, True, Softplus(), NormalPrior())}

    K = matern32_kernel(X1, X2, params)
    K_ref = reference_matern_kernel(X1, X2, 3.0 / 2, params)

    assert_allclose(K, K_ref)


@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
def test_matern52_kernel(dim, lengthscale):
    key = random.PRNGKey(2022)
    X1 = random.normal(key, shape=(10, dim))
    subkey, key = random.split(key)
    X2 = random.normal(subkey, shape=(20, dim))
    params = {"lengthscale": Parameter(lengthscale, True, Softplus(), NormalPrior())}

    K = matern52_kernel(X1, X2, params)
    K_ref = reference_matern_kernel(X1, X2, 5.0 / 2, params)

    assert_allclose(K, K_ref)


# ============================================================================
# Descriptor
# ============================================================================


def x_desc_jac():
    """
    x: (5, 1)
    desc: (5, 2)
    jac: (5, 2, 1)
    """
    # raw features
    x = jnp.linspace(-3, 3, 5).reshape(-1, 1)

    # descriptor (sin, cos)
    def d(x):
        return jnp.array([jnp.sin(x), jnp.cos(x)]).squeeze()

    desc = jax.vmap(d)(x)
    # jacobian
    dd_dx = jax.jacrev(d)
    jac = jax.vmap(dd_dx)(x)
    return x, desc, jac


# ============================================================================
# Test gradients/hessian kernels
# ============================================================================
#
# Here we test if the implementation of
#   d0k, d1k, d01k, d0kj, d1kj, d01kj
# of the kernel classes agree with the implementation given by automatic
# differentiation
#
# (Polynomial, no_intercept_polynomial_kernel_base), # TODO: take care of this


@pytest.mark.parametrize(
    "kernel, kernel_base",
    [
        (Constant, constant_kernel_base),
        (Linear, linear_kernel_base),
        (Polynomial, polynomial_kernel_base),
        (SquaredExponential, squared_exponential_kernel_base),
        # (Matern12, matern12_kernel_base), derivatives are not defined to Matern(1/2)
        (Matern32, matern32_kernel_base),
        (Matern52, matern52_kernel_base),
        (ExpSinSquared, expsinsquared_kernel_base),
    ],
)
@pytest.mark.parametrize("method", ["d0k", "d1k", "d01k"])
@pytest.mark.parametrize("active_dims", [None, jnp.array([0])])
def test_grads_k(kernel, kernel_base, method, active_dims):
    """
    Here we check that the d0k, d1k, d01k functions of each kernel
    yield the very same result as the d0k, d1k, d01k functions
    obtained by doing the automatic gradients of the kernel_base.
    For most kernels, the two functions will be the same: this check
    is here in order to ensure that custom d0k, d1k, d01k
    functions are correct.
    """
    x, desc, jac = x_desc_jac()
    # call the original d0k function of the kernel
    k = kernel(active_dims=active_dims)
    func = getattr(k, method)
    a = func(desc, desc, k.default_params())

    # create a new kernel without the overloaded d0k function
    class TestKernel(Kernel):
        def __init__(self, active_dims=None):
            self._kernel_base = kernel_base
            super().__init__(active_dims)

    tk = TestKernel(active_dims=active_dims)
    func = getattr(tk, method)
    b = func(desc, desc, k.default_params())
    assert_allclose(a, b)


@pytest.mark.parametrize(
    "kernel, kernel_base",
    [
        (Constant, constant_kernel_base),
        (Linear, linear_kernel_base),
        (Polynomial, polynomial_kernel_base),
        (SquaredExponential, squared_exponential_kernel_base),
        # (Matern12, matern12_kernel_base), derivatives are not defined for Matern(1/2)
        (Matern32, matern32_kernel_base),
        (Matern52, matern52_kernel_base),
        (ExpSinSquared, expsinsquared_kernel_base),
    ],
)
@pytest.mark.parametrize("method", ["d0kj", "d1kj"])
@pytest.mark.parametrize("active_dims", [None, jnp.array([0])])
def test_gradsjac_k(kernel, kernel_base, method, active_dims):
    """
    Here we check that the d0k, d1k, d01k functions of each kernel
    yield the very same result as the d0k, d1k, d01k functions
    obtained by doing the automatic gradients of the kernel_base.
    For most kernels, the two functions will be the same: this check
    is here in order to ensure that custom d0k, d1k, d01k
    functions are correct.
    """
    x, desc, jac = x_desc_jac()
    # call the original d0k function of the kernel
    k = kernel(active_dims=active_dims)
    func = getattr(k, method)
    a = func(desc, desc, k.default_params(), jac)

    # create a new kernel without the overloaded d0k function
    class TestKernel(Kernel):
        def __init__(self, active_dims=None):
            self._kernel_base = kernel_base
            super().__init__(active_dims)

    tk = TestKernel(active_dims)
    func = getattr(tk, method)
    b = func(desc, desc, k.default_params(), jac)
    assert_allclose(a, b)


@pytest.mark.parametrize(
    "kernel, kernel_base",
    [
        (Constant, constant_kernel_base),
        (Linear, linear_kernel_base),
        (Polynomial, polynomial_kernel_base),
        (SquaredExponential, squared_exponential_kernel_base),
        # (Matern12, matern12_kernel_base), derivatives are not defined for Matern(1/2)
        (Matern32, matern32_kernel_base),
        (Matern52, matern52_kernel_base),
        (ExpSinSquared, expsinsquared_kernel_base),
    ],
)
@pytest.mark.parametrize("active_dims", [None, jnp.array([0])])
def test_hessjac_k(kernel, kernel_base, active_dims):
    """
    Here we check that the d0k, d1k, d01k functions of each kernel
    yield the very same result as the d0k, d1k, d01k functions
    obtained by doing the automatic gradients of the kernel_base.
    For most kernels, the two functions will be the same: this check
    is here in order to ensure that custom d0k, d1k, d01k
    functions are correct.
    """
    x, desc, jac = x_desc_jac()
    # call the original function of the kernel
    k = kernel(active_dims=active_dims)
    a = k.d01kj(desc, desc, k.default_params(), jac, jac)

    # create a new kernel without the overloaded function
    class TestKernel(Kernel):
        def __init__(self, active_dims=None):
            self._kernel_base = kernel_base
            super().__init__(active_dims)

    tk = TestKernel(active_dims=active_dims)
    b = tk.d01kj(desc, desc, k.default_params(), jac, jac)
    assert_allclose(a, b)


# ============================================================================
# Test gradients/hessian kernels for Sum/Prod
# ============================================================================
#
# Here we test if the implementation of
#   d0k, d1k, d01k, d0kj, d1kj, d01kj
# of the Sum/Prod classes agree with the implementation given by automatic
# differentiation
#


@pytest.mark.parametrize(
    "kernel1, kernel1_base",
    [
        (Constant, constant_kernel_base),
        (Linear, linear_kernel_base),
        (Polynomial, polynomial_kernel_base),
        # (Polynomial, no_intercept_polynomial_kernel_base), # take care of this
        (SquaredExponential, squared_exponential_kernel_base),
        # (Matern12, matern12_kernel_base), derivatives are not defined for Matern(1/2)
        (Matern32, matern32_kernel_base),
        (Matern52, matern52_kernel_base),
        (ExpSinSquared, expsinsquared_kernel_base),
    ],
)
@pytest.mark.parametrize(
    "kernel2, kernel2_base",
    [
        (Constant, constant_kernel_base),
        (Linear, linear_kernel_base),
        (Polynomial, polynomial_kernel_base),
        # (Polynomial, no_intercept_polynomial_kernel_base), # take care of this
        (SquaredExponential, squared_exponential_kernel_base),
        # (Matern12, matern12_kernel_base), derivatives are not defined for Matern(1/2)
        (Matern32, matern32_kernel_base),
        (Matern52, matern52_kernel_base),
        (ExpSinSquared, expsinsquared_kernel_base),
    ],
)
@pytest.mark.parametrize("method", ["d0k", "d1k", "d01k"])
@pytest.mark.parametrize("op, op_base", [(Sum, sum_kernels), (Prod, prod_kernels)])
@pytest.mark.parametrize(
    "active_dims1, active_dims2", [(None, None), (jnp.array([0]), jnp.array([1]))]
)
def test_grads_sumprod(
    kernel1,
    kernel1_base,
    kernel2,
    kernel2_base,
    method,
    op,
    op_base,
    active_dims1,
    active_dims2,
):
    """
    Here we check that the d0k, d1k, d01k functions of each kernel
    yield the very same result as the d0k, d1k, d01k functions
    obtained by doing the automatic gradients of the kernel_base.
    For most kernels, the two functions will be the same: this check
    is here in order to ensure that custom d0k, d1k, d01k
    functions are correct.
    """
    x, desc, jac = x_desc_jac()
    # call the original d0k function of the kernel
    k1 = kernel1(active_dims=active_dims1)
    k2 = kernel2(active_dims=active_dims2)
    k = op(k1, k2)
    func = getattr(k, method)
    a = func(desc, desc, k.default_params())

    # create a new kernel without the overloaded d0k function
    class TestKernel(Kernel):
        def __init__(self, active_dims=None):
            self._kernel_base = op_base(k1._kernel_base, k2._kernel_base)
            super().__init__(active_dims)

    tk = TestKernel()
    func = getattr(tk, method)
    b = func(desc, desc, k.default_params())
    assert_allclose(a, b)


@pytest.mark.parametrize(
    "kernel1, kernel1_base",
    [
        (Constant, constant_kernel_base),
        (Linear, linear_kernel_base),
        (Polynomial, polynomial_kernel_base),
        # (Polynomial, no_intercept_polynomial_kernel_base), # take care of this
        (SquaredExponential, squared_exponential_kernel_base),
        # (Matern12, matern12_kernel_base), derivatives are not defined for Matern(1/2)
        (Matern32, matern32_kernel_base),
        (Matern52, matern52_kernel_base),
        (ExpSinSquared, expsinsquared_kernel_base),
    ],
)
@pytest.mark.parametrize(
    "kernel2, kernel2_base",
    [
        (Constant, constant_kernel_base),
        (Linear, linear_kernel_base),
        (Polynomial, polynomial_kernel_base),
        # (Polynomial, no_intercept_polynomial_kernel_base), # take care of this
        (SquaredExponential, squared_exponential_kernel_base),
        # (Matern12, matern12_kernel_base), derivatives are not defined for Matern(1/2)
        (Matern32, matern32_kernel_base),
        (Matern52, matern52_kernel_base),
        (ExpSinSquared, expsinsquared_kernel_base),
    ],
)
@pytest.mark.parametrize(
    "active_dims1, active_dims2", [(None, None), (jnp.array([0]), jnp.array([1]))]
)
def test_gradsjac_sumprod(
    kernel1,
    kernel1_base,
    kernel2,
    kernel2_base,
    active_dims1,
    active_dims2,
):
    """
    Here we check that the d0k, d1k, d01k functions of each kernel
    yield the very same result as the d0k, d1k, d01k functions
    obtained by doing the automatic gradients of the kernel_base.
    For most kernels, the two functions will be the same: this check
    is here in order to ensure that custom d0k, d1k, d01k
    functions are correct.
    """
    x, desc, jac = x_desc_jac()

    k1 = kernel1(active_dims=active_dims1)
    k2 = kernel2(active_dims=active_dims2)

    # # Sum
    # k = Sum(k1, k2)

    # # create a new kernel without the overloaded d0k function
    # class TestKernel(Kernel):
    #     def __init__(self, active_dims=None):
    #         self._kernel_base = sum_kernels(k1._kernel_base, k2._kernel_base)
    #         super().__init__(active_dims)

    # tk = TestKernel()

    # a = k.d0kj(desc, desc, k.default_params(), jac)
    # b = tk.d0kj(desc, desc, k.default_params(), jac)
    # # sometimes we have a 1e-18 instead of a plain zero
    # # but I would not consider this to be an error
    # assert_allclose(a, b, rtol=1., atol=1e-15)

    # a = k.d1kj(desc, desc, k.default_params(), jac)
    # b = tk.d1kj(desc, desc, k.default_params(), jac)
    # assert_allclose(a, b, rtol=1., atol=1e-15)

    # a = k.d01kj(desc, desc, k.default_params(), jac, jac)
    # b = tk.d01kj(desc, desc, k.default_params(), jac, jac)
    # assert_allclose(a, b, rtol=1., atol=1e-15)

    # Prod
    k = Prod(k1, k2)

    # create a new kernel without the overloaded d0k function
    class TestKernel(Kernel):
        def __init__(self, active_dims=None):
            self._kernel_base = prod_kernels(k1._kernel_base, k2._kernel_base)
            super().__init__(active_dims)

    tk = TestKernel()

    a = k.d0kj(desc, desc, k.default_params(), jac)
    b = tk.d0kj(desc, desc, k.default_params(), jac)
    # sometimes we have a 1e-18 instead of a plain zero
    # but I would not consider this to be an error
    assert_allclose(a, b, rtol=1.0, atol=1e-15)

    a = k.d1kj(desc, desc, k.default_params(), jac)
    b = tk.d1kj(desc, desc, k.default_params(), jac)
    assert_allclose(a, b, rtol=1.0, atol=1e-15)

    a = k.d01kj(desc, desc, k.default_params(), jac, jac)
    b = tk.d01kj(desc, desc, k.default_params(), jac, jac)
    assert_allclose(a, b, rtol=1.0, atol=1e-15)


# ============================================================================
# Active dimensions
# ============================================================================


@pytest.mark.parametrize(
    "kernel",
    [
        Constant,
        Linear,
        Polynomial,
        # (Polynomial, no_intercept_polynomial_kernel_base), # take care of this
        SquaredExponential,
        Matern12,
        Matern32,
        Matern52,
        ExpSinSquared,
    ],
)
@pytest.mark.parametrize("active_dims", [None, jnp.array([0]), jnp.array([1])])
def test_active_dims(kernel, active_dims):
    x, desc, jac = x_desc_jac()
    # kernel operating on active dimensions
    k_sliced = kernel(active_dims=active_dims)
    # kernel operating on every dimensions
    k_full = kernel(active_dims=None)
    # truncating before feeding the input to the kernel
    # must be the same as using the active dimensions
    a = k_sliced(desc, desc, k_sliced.default_params())
    if active_dims is None:
        desc_sliced = desc
    else:
        desc_sliced = desc[:, active_dims]
    b = k_full(desc_sliced, desc_sliced, k_full.default_params())
    assert_allclose(a, b)


@pytest.mark.parametrize(
    "kernel",
    [
        Constant,
        Linear,
        Polynomial,
        # (Polynomial, no_intercept_polynomial_kernel_base), # take care of this
        SquaredExponential,
        Matern12,
        Matern32,
        Matern52,
        ExpSinSquared,
    ],
)
@pytest.mark.parametrize("active_dims", [jnp.array([0]), jnp.array([1])])
def test_grads_active_dims(kernel, active_dims):
    # NOTE: this is hardcoded to work with a descriptor that is 2-dimensional
    x, desc, jac = x_desc_jac()
    k = kernel(active_dims=active_dims)

    # derivative w.r.t first argument
    a = k.d0k(desc, desc, k.default_params())
    # check that we have zeros in the correct place
    a = jnp.take(a, jnp.arange(a.shape[0])[1 - active_dims[0] :: 2], axis=0)
    assert_array_equal(a, 0.0)

    # derivative w.r.t. second argument
    a = k.d1k(desc, desc, k.default_params())
    a = jnp.take(a, jnp.arange(a.shape[1])[1 - active_dims[0] :: 2], axis=1)
    assert_array_equal(a, 0.0)

    # hessian
    a = k.d01k(desc, desc, k.default_params())
    a_ = jnp.take(a, jnp.arange(a.shape[0])[1 - active_dims[0] :: 2], axis=0)
    assert_array_equal(a_, 0.0)
    a_ = jnp.take(a, jnp.arange(a.shape[1])[1 - active_dims[1] :: 2], axis=1)
    assert_array_equal(a_, 0.0)


@pytest.mark.parametrize(
    "kernel",
    [
        Constant,
        Linear,
        Polynomial,
        # (Polynomial, no_intercept_polynomial_kernel_base), # take care of this
        SquaredExponential,
        Matern12,
        Matern32,
        Matern52,
        ExpSinSquared,
    ],
)
@pytest.mark.parametrize("active_dims", [jnp.array([0]), jnp.array([1])])
def test_gradsjac_active_dims(kernel, active_dims):
    """
    We check that computing the product explicitly between the d0k etc
    and the jacobian provides the same result as directly calling d0kj etc.
    """
    x, desc, jac = x_desc_jac()
    ns, nf = desc.shape

    k = kernel(active_dims=active_dims)

    # derivative w.r.t first argument
    a = k.d0k(desc, desc, k.default_params())
    out = []
    for i in range(ns):
        a_block = a[i * nf : (i * nf + nf)]
        res = jnp.einsum("ij,il->lj", a_block, jac[i])
        out.append(res)
    out = jnp.concatenate(out, axis=0)
    b = k.d0kj(desc, desc, k.default_params(), jac)
    assert_allclose(out, b)

    # derivative w.r.t. second argument
    a = k.d1k(desc, desc, k.default_params())
    out = []
    for i in range(ns):
        a_block = a[:, i * nf : (i * nf + nf)]
        res = jnp.einsum("ji,il->jl", a_block, jac[i])
        out.append(res)
    out = jnp.concatenate(out, axis=1)
    b = k.d1kj(desc, desc, k.default_params(), jac)
    assert_allclose(out, b)

    # hessian
    a = k.d01k(desc, desc, k.default_params())
    out = []
    for i in range(ns):
        outr = []
        for j in range(ns):
            a_block = a[i * nf : (i * nf + nf)][:, j * nf : (j * nf + nf)]
            res = jnp.einsum("il,ij,jm->lm", jac[i], a_block, jac[j])
            outr.append(res)
        outr = jnp.concatenate(outr, axis=1)
        out.append(outr)
    out = jnp.concatenate(out, axis=0)
    b = k.d01kj(desc, desc, k.default_params(), jac, jac)
    assert_allclose(out, b)
