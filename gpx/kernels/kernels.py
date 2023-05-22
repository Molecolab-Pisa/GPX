from typing import Dict

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from ..parameters.parameter import Parameter
from ..priors import NormalPrior
from ..utils import euclidean_distance, inverse_softplus, softplus, squared_distances
from .kernelizers import grad_kernelize, kernelize
from .operations import (
    prod_kernels,
    prod_kernels_deriv,
    prod_kernels_deriv01,
    prod_kernels_deriv01_jac,
    prod_kernels_deriv_jac,
    sum_kernels,
    sum_kernels_jac,
    sum_kernels_jac2,
)

# =============================================================================
# Constant Kernel
# =============================================================================


def _constant_kernel_base(x1: ArrayLike, x2: ArrayLike, variance: float) -> Array:
    return variance


@jit
def constant_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    variance = params["variance"].value
    return _constant_kernel_base(x1, x2, variance)


def _constant_kernel(x1: ArrayLike, x2: ArrayLike, variance: float) -> Array:
    ns1, _ = x1.shape
    ns2, _ = x2.shape
    return jnp.ones((ns1, ns2), dtype=jnp.float64) * variance


@jit
def constant_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    variance = params["variance"].value
    return _constant_kernel(x1, x2, variance)


# =============================================================================
# Linear Kernel
# =============================================================================


def _linear_kernel_base(x1: ArrayLike, x2: ArrayLike) -> Array:
    return jnp.dot(x1, x2)


@jit
def linear_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    return _linear_kernel_base(x1, x2)


def _linear_kernel(x1: ArrayLike, x2: ArrayLike) -> Array:
    return x1 @ x2.T


@jit
def linear_kernel(x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]) -> Array:
    return _linear_kernel(x1, x2)


# =============================================================================
# Squared Exponential Kernel
# =============================================================================


def _squared_exponential_kernel_base(
    x1: ArrayLike, x2: ArrayLike, lengthscale: float
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    return jnp.exp(-jnp.sum((z1 - z2) ** 2))


@jit
def squared_exponential_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _squared_exponential_kernel_base(x1, x2, lengthscale)


def _squared_exponential_kernel(
    x1: ArrayLike, x2: ArrayLike, lengthscale: float
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    return jnp.exp(-d2)


@jit
def squared_exponential_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _squared_exponential_kernel(x1, x2, lengthscale)


# =============================================================================
# Matern(1/2) Kernel
# =============================================================================


def _matern12_kernel_base(x1: ArrayLike, x2: ArrayLike, lengthscale: float) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d = euclidean_distance(z1, z2)
    return jnp.exp(-d)


@jit
def matern12_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern12_kernel_base(x1, x2, lengthscale)


def _matern12_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: float) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    d = jnp.sqrt(jnp.maximum(d2, 1e-36))
    return jnp.exp(-d)


@jit
def matern12_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern12_kernel(x1, x2, lengthscale)


# =============================================================================
# Matern(3/2) Kernel
# =============================================================================


def _matern32_kernel_base(x1: ArrayLike, x2: ArrayLike, lengthscale: float) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d = jnp.sqrt(3.0) * euclidean_distance(z1, z2)
    return (1.0 + d) * jnp.exp(-d)


@jit
def matern32_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern32_kernel_base(x1, x2, lengthscale)


def _matern32_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: float) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    d = jnp.sqrt(3.0) * jnp.sqrt(jnp.maximum(d2, 1e-36))
    return (1.0 + d) * jnp.exp(-d)


@jit
def matern32_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern32_kernel(x1, x2, lengthscale)


# =============================================================================
# Matern(5/2) Kernel
# =============================================================================


def _matern52_kernel_base(x1: ArrayLike, x2: ArrayLike, lengthscale: float) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d = jnp.sqrt(5.0) * euclidean_distance(z1, z2)
    return (1.0 + d + d**2 / 3.0) * jnp.exp(-d)


@jit
def matern52_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern52_kernel_base(x1, x2, lengthscale)


def _matern52_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: float) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    d = jnp.sqrt(5.0) * jnp.sqrt(jnp.maximum(d2, 1e-36))
    return (1.0 + d + d**2 / 3.0) * jnp.exp(-d)


@jit
def matern52_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern52_kernel(x1, x2, lengthscale)


# =============================================================================
# Kernel aliases
# =============================================================================

const_kernel = constant_kernel
lin_kernel = linear_kernel
se_kernel = squared_exponential_kernel
m12_kernel = matern12_kernel
m32_kernel = matern32_kernel
m52_kernel = matern52_kernel


# =============================================================================
# Classes
# =============================================================================


class Kernel:
    """base class representing a kernel

    This class is a thin wrapper around kernel function.
    In principle, to implement a fully working kernel with
    the minimum effort: (i) define a `kernel_base` function,
    which computes the kernel between **two** samples (i.e., not
    on batches of data). (2) write the corresponding kernel
    class that inherits from this class, and that calls
    super().__init__() **after** specifying the base kernel, e.g.:

    >>> def my_custom_kernel_base(x1: ArrayLike, x2: ArrayLike, params: Dict):
    ...     # write your implementation here
    >>>
    >>> class MyCustomKernel(Kernel):
    ...     def __init__(self):
    ...         self._kernel_base = my_custom_kernel_base
    ...         super().__init__()

    Classes defined in this way will have automatically defined
    the following operations working for batches of samples,
    in the form of callables:
    *   kernel function (self.k)
    *   derivative kernel wrt first argument (self.d0k)
    *   derivative kernel wrt second argument (self.d1k)
    *   hessian kernel (self.d01k)
    *   derivative kernel - jacobian product wrt first argument (self.d0kj)
    *   derivative kernel - jacobian product wrt second argument (self.d1kj)
    *   hessian kernel - jacobian product (self.d01kj)

    In addition, calling the class will evaluate the kernel function, i.e.,
    it is equivalent of calling `self.k`.

    If you have an implementation of some of these functions that you want
    to use instead of the default one, simply define it after initializing
    the parent class:

    >>> def my_custom_kernel_base(x1: ArrayLike, x2: ArrayLike, params: Dict):
    ...     # write your implementation here
    >>>
    >>> class MyCustomKernel(Kernel):
    ...     def __init__(self):
    ...         self._kernel_base = my_custom_kernel_base
    ...         super().__init__()
    ...         # use a custom faster version for evaluating the kernel and the hessian
    ...         self.k = my_custom_faster_kernel
    ...         self.d01k = my_custom_faster_hessian_kernel
    """

    def __init__(self) -> None:
        # kernel
        self.k = kernelize(self._kernel_base)

        # derivative/hessian kernel
        self.d0k = grad_kernelize(argnums=0, with_jacob=False)(self._kernel_base)
        self.d1k = grad_kernelize(argnums=1, with_jacob=False)(self._kernel_base)
        self.d01k = grad_kernelize(argnums=(0, 1), with_jacob=False)(self._kernel_base)

        # derivative/hessian kernel-jacobian products
        self.d0kj = grad_kernelize(argnums=0, with_jacob=True)(self._kernel_base)
        self.d1kj = grad_kernelize(argnums=1, with_jacob=True)(self._kernel_base)
        self.d01kj = grad_kernelize(argnums=(0, 1), with_jacob=True)(self._kernel_base)

    def __call__(self, x1: ArrayLike, x2: ArrayLike, params: Dict) -> Array:
        return self.k(x1, x2, params)


class Constant(Kernel):
    def __init__(self) -> None:
        self._kernel_base = constant_kernel_base
        super().__init__()
        # faster version for evaluating k
        self.k = constant_kernel

    def default_params(self):
        return dict(
            variance=Parameter(
                value=1.0,
                trainable=True,
                forward_transform=softplus,
                backward_transform=inverse_softplus,
                prior=NormalPrior(loc=0.0, scale=1.0),
            )
        )


class Linear(Kernel):
    def __init__(self) -> None:
        self._kernel_base = linear_kernel_base
        super().__init__()
        # faster version for evaluating k
        self.k = linear_kernel

    def default_params(self):
        return dict()


class SquaredExponential(Kernel):
    def __init__(self) -> None:
        self._kernel_base = squared_exponential_kernel_base
        super().__init__()
        # faster version for evaluating k
        self.k = squared_exponential_kernel

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                forward_transform=softplus,
                backward_transform=inverse_softplus,
                prior=NormalPrior(loc=0.0, scale=1.0),
            )
        )


class Matern12(Kernel):
    def __init__(self) -> None:
        self._kernel_base = matern12_kernel_base
        super().__init__()
        # faster version for evaluating k
        self.k = matern12_kernel

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                forward_transform=softplus,
                backward_transform=inverse_softplus,
                prior=NormalPrior(loc=0.0, scale=1.0),
            )
        )


class Matern32(Kernel):
    def __init__(self) -> None:
        self._kernel_base = matern32_kernel_base
        super().__init__()
        # faster version for evaluating k
        self.k = matern32_kernel

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                forward_transform=softplus,
                backward_transform=inverse_softplus,
                prior=NormalPrior(loc=0.0, scale=1.0),
            )
        )


class Matern52(Kernel):
    def __init__(self) -> None:
        self._kernel_base = matern52_kernel_base
        super().__init__()
        # faster version for evaluating k
        self.k = matern52_kernel

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                forward_transform=softplus,
                backward_transform=inverse_softplus,
                prior=NormalPrior(loc=0.0, scale=1.0),
            )
        )


class Sum:
    """Class representing the sum of two kernels

    *   kernel function (self.k)
    *   derivative kernel wrt first argument (self.d0k)
    *   derivative kernel wrt second argument (self.d1k)
    *   hessian kernel (self.d01k)
    *   derivative kernel - jacobian product wrt first argument (self.d0kj)
    *   derivative kernel - jacobian product wrt second argument (self.d1kj)
    *   hessian kernel - jacobian product (self.d01kj)

    In addition, calling the class will evaluate the kernel function, i.e.,
    it is equivalent of calling `self.k`.

    Parameters for both kernels must be passed as
    {'kernel1' : params1, 'kernel2' : params2},
    where params1 and params2 are standard GPX parameters for kernels.
    """

    def __init__(self, kernel1: Kernel, kernel2: Kernel) -> None:
        # kernel
        self._kernel_base = sum_kernels(kernel1._kernel_base, kernel2._kernel_base)
        self.k = sum_kernels(kernel1.k, kernel2.k)
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        # derivative/hessian kernel
        self.d0k = sum_kernels(kernel1.d0k, kernel2.d0k)
        self.d1k = sum_kernels(kernel1.d1k, kernel2.d1k)
        self.d01k = sum_kernels(kernel1.d01k, kernel2.d01k)

        # derivative/hessian kernel-jacobian products
        self.d0kj = sum_kernels_jac(kernel1.d0kj, kernel2.d0kj)
        self.d1kj = sum_kernels_jac(kernel1.d1kj, kernel2.d1kj)
        self.d01kj = sum_kernels_jac2(kernel1.d01kj, kernel2.d01kj)

    def __call__(self, x1: ArrayLike, x2: ArrayLike, params: Dict) -> Array:
        return self.k(x1, x2, params)

    def default_params(self):
        # simply delegate
        return {
            "kernel1": self.kernel1.default_params(),
            "kernel2": self.kernel2.default_params(),
        }


class Prod:
    """Class representing the Schur (element-wise) product of two kernels

    *   kernel function (self.k)
    *   derivative kernel wrt first argument (self.d0k)
    *   derivative kernel wrt second argument (self.d1k)
    *   hessian kernel (self.d01k)
    *   derivative kernel - jacobian product wrt first argument (self.d0kj)
    *   derivative kernel - jacobian product wrt second argument (self.d1kj)
    *   hessian kernel - jacobian product (self.d01kj)

    In addition, calling the class will evaluate the kernel function, i.e.,
    it is equivalent of calling `self.k`.

    Parameters for both kernels must be passed as
    {'kernel1' : params1, 'kernel2' : params2},
    where params1 and params2 are standard GPX parameters for kernels.
    """

    def __init__(self, kernel1: Kernel, kernel2: Kernel) -> None:
        # kernel
        self._kernel_base = prod_kernels(kernel1._kernel_base, kernel2._kernel_base)
        self.k = prod_kernels(kernel1.k, kernel2.k)
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        # derivative/hessian kernel
        self.d0k = prod_kernels_deriv(kernel1.k, kernel2.k, kernel1.d0k, kernel2.d0k, 0)
        self.d1k = prod_kernels_deriv(
            kernel1.k, kernel2.k, kernel1.d1k, kernel2.d1k, -1
        )
        self.d01k = prod_kernels_deriv01(
            kernel1.k,
            kernel2.k,
            kernel1.d0k,
            kernel2.d0k,
            kernel1.d1k,
            kernel2.d1k,
            kernel1.d01k,
            kernel2.d01k,
        )

        # derivative/hessian kernel-jacobian products
        self.d0kj = prod_kernels_deriv_jac(
            kernel1.k, kernel2.k, kernel1.d0kj, kernel2.d0kj, 0
        )
        self.d1kj = prod_kernels_deriv_jac(
            kernel1.k, kernel2.k, kernel1.d1kj, kernel2.d1kj, -1
        )
        self.d01kj = prod_kernels_deriv01_jac(
            kernel1.k,
            kernel2.k,
            kernel1.d0kj,
            kernel2.d0kj,
            kernel1.d1kj,
            kernel2.d1kj,
            kernel1.d01kj,
            kernel2.d01kj,
        )

    def __call__(self, x1: ArrayLike, x2: ArrayLike, params: Dict) -> Array:
        return self.k(x1, x2, params)

    def default_params(self):
        # simply delegate
        return {
            "kernel1": self.kernel1.default_params(),
            "kernel2": self.kernel2.default_params(),
        }
