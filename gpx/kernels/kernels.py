from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, custom_jvp, jit
from jax.typing import ArrayLike

from ..bijectors import Identity, Softplus
from ..parameters.parameter import Parameter
from ..priors import GammaPrior, NormalPrior
from ..utils import squared_distances
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
# Kernel functions
# =============================================================================

# Before the kernel classes, there we have a series of kernel functions.
# The actual implementation of the kernels is inside these functions, and the
# kernel classes act mostly as a wrapper.
# The kernel functions accept the "active_dims" argument. When used inside the
# kernel_base functions, which, in combination with the kernelizers, define all
# the kernel functions (gradients, hessians), this ensures that the shape of
# a jacobian or hessian kernel is the correct one, with zeros in correspondence
# to unused features. If instead you want to implement your own, advanced version
# of jacobian/hessian kernel, pay attention to put the zeros in the correct places
# when handling the "active_dims".
# A note to avoid confusions: checking if a variable "is None" is perfectly allowed
# by JIT as it can be evaluated at tracing time. This is used to avoid the need
# of speciying the active dimensions when all the dimensions are requested.
# Also note that inside the kernel_base functions, x has shape (n_features,),
# while instead in the kernel functions x has shape (n_samples, n_features)

# =============================================================================
# Constant Kernel
# =============================================================================


def _constant_kernel_base(x1: ArrayLike, x2: ArrayLike, variance: ArrayLike) -> Array:
    return variance


@jit
def constant_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    variance = params["variance"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _constant_kernel_base(x1[active_dims], x2[active_dims], variance)


def _constant_kernel(x1: ArrayLike, x2: ArrayLike, variance: ArrayLike) -> Array:
    ns1, _ = x1.shape
    ns2, _ = x2.shape
    return jnp.ones((ns1, ns2), dtype=jnp.float64) * variance


@jit
def constant_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    variance = params["variance"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _constant_kernel(x1[:, active_dims], x2[:, active_dims], variance)


# =============================================================================
# Linear Kernel
# =============================================================================


def _linear_kernel_base(x1: ArrayLike, x2: ArrayLike) -> Array:
    return jnp.dot(x1, x2)


@jit
def linear_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _linear_kernel_base(x1[active_dims], x2[active_dims])


def _linear_kernel(x1: ArrayLike, x2: ArrayLike) -> Array:
    return x1 @ x2.T


@jit
def linear_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _linear_kernel(x1[:, active_dims], x2[:, active_dims])


# =============================================================================
# Polynomial Kernel
# =============================================================================


def _polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return (offset + jnp.dot(x1, x2)) ** degree


@jit
def polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _polynomial_kernel_base(x1[active_dims], x2[active_dims], degree, offset)


def _polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return (offset + x1 @ x2.T) ** degree


@jit
def polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _polynomial_kernel(x1[:, active_dims], x2[:, active_dims], degree, offset)


# =============================================================================
# No Intercept Polynomial Kernel
# =============================================================================


def _no_intercept_polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return ((offset + jnp.dot(x1, x2)) ** degree) - (offset**degree)


@jit
def no_intercept_polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _no_intercept_polynomial_kernel_base(
        x1[active_dims], x2[active_dims], degree, offset
    )


def _no_intercept_polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return ((offset + x1 @ x2.T) ** degree) - (offset**degree)


@jit
def no_intercept_polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _no_intercept_polynomial_kernel(
        x1[:, active_dims], x2[:, active_dims], degree, offset
    )


# =============================================================================
# Squared Exponential Kernel
# =============================================================================


def _squared_exponential_kernel_base(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    return jnp.exp(-jnp.sum((z1 - z2) ** 2))


@jit
def squared_exponential_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _squared_exponential_kernel_base(
        x1[active_dims], x2[active_dims], lengthscale
    )


def _squared_exponential_kernel(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    return jnp.exp(-d2)


@jit
def squared_exponential_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _squared_exponential_kernel(
        x1[:, active_dims], x2[:, active_dims], lengthscale
    )


# =============================================================================
# Matern(1/2) Kernel
# =============================================================================


def _matern12_kernel_base(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = jnp.sum((z1 - z2) ** 2)
    d = jnp.sqrt(jnp.maximum(d2, 1e-36))
    return jnp.exp(-d)


@jit
def matern12_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _matern12_kernel_base(x1[active_dims], x2[active_dims], lengthscale)


def _matern12_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    d = jnp.sqrt(jnp.maximum(d2, 1e-36))
    return jnp.exp(-d)


@jit
def matern12_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _matern12_kernel(x1[:, active_dims], x2[:, active_dims], lengthscale)


# =============================================================================
# Matern(3/2) Kernel
# =============================================================================


@custom_jvp
def _matern32_kernel_base(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = 3.0 * jnp.sum((z1 - z2) ** 2)
    d = jnp.sqrt(jnp.maximum(d2, 1e-36))
    return (1.0 + d) * jnp.exp(-d)


@jit
def matern32_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _matern32_kernel_base(x1[active_dims], x2[active_dims], lengthscale)


def _matern32_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    d = jnp.sqrt(3.0) * jnp.sqrt(jnp.maximum(d2, 1e-36))
    return (1.0 + d) * jnp.exp(-d)


@jit
def matern32_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _matern32_kernel(x1[:, active_dims], x2[:, active_dims], lengthscale)


def _matern32_kernel_base_t0(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    diff = jnp.sqrt(3.0) * (z1 - z2)
    d = jnp.sqrt(jnp.maximum(jnp.sum(diff**2), 1e-36))
    return -jnp.sqrt(3.0) / lengthscale * jnp.exp(-d) * diff


def _matern32_kernel_base_t1(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    return -_matern32_kernel_base_t0(x1, x2, lengthscale)


def _matern32_kernel_base_t2(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    diff = jnp.sqrt(3.0) * (z1 - z2)
    d = jnp.sqrt(jnp.maximum(jnp.sum(diff**2), 1e-36))
    return 1.0 / lengthscale * jnp.exp(-d) * d**2


_matern32_kernel_base.defjvps(
    lambda x1_dot, primal_out, x1, x2, lengthscale: (
        _matern32_kernel_base_t0(x1, x2, lengthscale) @ x1_dot
    ).reshape(primal_out.shape),
    lambda x2_dot, primal_out, x1, x2, lengthscale: (
        _matern32_kernel_base_t1(x1, x2, lengthscale) @ x2_dot
    ).reshape(primal_out.shape),
    lambda lengthscale_dot, primal_out, x1, x2, lengthscale: (
        jnp.atleast_1d(_matern32_kernel_base_t2(x1, x2, lengthscale))
        @ jnp.atleast_1d(lengthscale_dot)
    ).reshape(primal_out.shape),
)


# =============================================================================
# Matern(5/2) Kernel
# =============================================================================


@custom_jvp
def _matern52_kernel_base(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = 5.0 * jnp.sum((z1 - z2) ** 2)
    d = jnp.sqrt(jnp.maximum(d2, 1e-36))
    return (1.0 + d + d2 / 3.0) * jnp.exp(-d)


@jit
def matern52_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _matern52_kernel_base(x1[active_dims], x2[active_dims], lengthscale)


def _matern52_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    d = jnp.sqrt(5.0) * jnp.sqrt(jnp.maximum(d2, 1e-36))
    return (1.0 + d + d**2 / 3.0) * jnp.exp(-d)


@jit
def matern52_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[1])
    return _matern52_kernel(x1[:, active_dims], x2[:, active_dims], lengthscale)


def _matern52_kernel_base_t0(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    diff = jnp.sqrt(5.0) * (z1 - z2)
    d = jnp.sqrt(jnp.maximum(jnp.sum(diff**2), 1e-36))
    return -(jnp.sqrt(5.0) / (3.0 * lengthscale)) * (1 + d) * jnp.exp(-d) * diff


def _matern52_kernel_base_t1(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    return -_matern52_kernel_base_t0(x1, x2, lengthscale)


def _matern52_kernel_base_t2(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> Array:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    diff = jnp.sqrt(5.0) * (z1 - z2)
    d = jnp.sqrt(jnp.maximum(jnp.sum(diff**2), 1e-36))
    return 1.0 / (3.0 * lengthscale) * jnp.exp(-d) * d**2 * (1 + d)


_matern52_kernel_base.defjvps(
    lambda x1_dot, primal_out, x1, x2, lengthscale: (
        _matern52_kernel_base_t0(x1, x2, lengthscale) @ x1_dot
    ).reshape(primal_out.shape),
    lambda x2_dot, primal_out, x1, x2, lengthscale: (
        _matern52_kernel_base_t1(x1, x2, lengthscale) @ x2_dot
    ).reshape(primal_out.shape),
    lambda lengthscale_dot, primal_out, x1, x2, lengthscale: (
        jnp.atleast_1d(_matern52_kernel_base_t2(x1, x2, lengthscale))
        @ jnp.atleast_1d(lengthscale_dot)
    ).reshape(primal_out.shape),
)

# =============================================================================
# Periodic kernels
# =============================================================================


def _expsinsquared_kernel_base(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike, period: ArrayLike
) -> Array:
    return jnp.exp(
        -4 * jnp.sum(jnp.sin(jnp.pi * jnp.abs(x1 - x2) / period) ** 2 / lengthscale)
    )


@jit
def expsinsquared_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], active_dims=None
) -> Array:
    lengthscale = params["lengthscale"].value
    period = params["periodicity"].value
    if active_dims is None:
        active_dims = jnp.arange(x1.shape[0])
    return _expsinsquared_kernel_base(
        x1[active_dims], x2[active_dims], lengthscale, period
    )


# =============================================================================
# Classes
# =============================================================================


class Kernel:
    """base class representing a kernel

    This class is a thin wrapper around kernel function.
    In principle, to implement a fully working kernel with
    the minimum effort: (1) define a `kernel_base` function,
    which computes the kernel between **two** samples (i.e., not
    on batches of data). (2) write the corresponding kernel
    class that inherits from this class, and that calls
    super().__init__() **after** specifying the base kernel, e.g.:

    >>> def my_custom_kernel_base(x1: ArrayLike, x2: ArrayLike, params: Dict):
    ...     # write your implementation here
    >>>
    >>> class MyCustomKernel(Kernel):
    ...     def __init__(self, active_dims):
    ...         self._kernel_base = my_custom_kernel_base
    ...         super().__init__(active_dims)

    'active_dims' specifies the subset of features to include when
    evaluating the kernel function. By default all the features are included.

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
    ...     def __init__(self, active_dims):
    ...         self._kernel_base = my_custom_kernel_base
    ...         super().__init__(active_dims)
    ...         # use a custom faster version for evaluating the kernel and the hessian
    ...         self.k = self.filter_input(my_custom_faster_kernel)
    ...         self.d01k = self.filter_input(my_custom_faster_hessian_kernel)
    """

    def __init__(self, active_dims: ArrayLike = None) -> None:
        # kernel
        self.active_dims = active_dims

        self.k = partial(kernelize(self._kernel_base), active_dims=active_dims)

        # derivative/hessian kernel
        self.d0k = partial(
            grad_kernelize(argnums=0, with_jacob=False)(self._kernel_base),
            active_dims=active_dims,
        )
        self.d1k = partial(
            grad_kernelize(argnums=1, with_jacob=False)(self._kernel_base),
            active_dims=active_dims,
        )
        self.d01k = partial(
            grad_kernelize(argnums=(0, 1), with_jacob=False)(self._kernel_base),
            active_dims=active_dims,
        )

        # derivative/hessian kernel-jacobian products
        self.d0kj = partial(
            grad_kernelize(argnums=0, with_jacob=True)(self._kernel_base),
            active_dims=active_dims,
        )
        self.d1kj = partial(
            grad_kernelize(argnums=1, with_jacob=True)(self._kernel_base),
            active_dims=active_dims,
        )
        self.d01kj = partial(
            grad_kernelize(argnums=(0, 1), with_jacob=True)(self._kernel_base),
            active_dims=active_dims,
        )

    def __call__(self, x1: ArrayLike, x2: ArrayLike, params: Dict) -> Array:
        return self.k(x1, x2, params)

    def __add__(self, k: "Kernel") -> "Sum":
        return Sum(self, k)

    def __mul__(self, k: "Kernel") -> "Prod":
        return Prod(self, k)


class Constant(Kernel):
    """Constant kernel

    The Constant kernel:

        k(x, x') = v

    where v is the variance.
    """

    def __init__(self, active_dims: ArrayLike = None) -> None:
        self._kernel_base = constant_kernel_base
        super().__init__(active_dims)
        # faster version for evaluating k
        self.k = partial(constant_kernel, active_dims=active_dims)

    def default_params(self):
        return dict(
            variance=Parameter(
                value=1.0,
                trainable=True,
                bijector=Softplus(),
                prior=GammaPrior(),
            )
        )


class Linear(Kernel):
    """Linear kernel

    The Linear kernel:

        k(x, x') = x∙x'

    """

    def __init__(self, active_dims: ArrayLike = None) -> None:
        self._kernel_base = linear_kernel_base
        super().__init__(active_dims)
        # faster version for evaluating k
        self.k = partial(linear_kernel, active_dims=active_dims)

    def default_params(self):
        return dict()


class Polynomial(Kernel):
    """Polynomial kernel

    The Polynomial kernel:

        k(x, x') = (c + x∙x')^(d)

    where c is the offset and d is the degree.

    If 'no_intercept' is specified, the constant term is subtracted:

        k(x, x') = (c + x∙x')^(d) - (c)^(d)
    """

    def __init__(
        self, active_dims: ArrayLike = None, no_intercept: bool = False
    ) -> None:
        if no_intercept:
            self._kernel_base = no_intercept_polynomial_kernel_base
        else:
            self._kernel_base = polynomial_kernel_base
        super().__init__(active_dims)
        # faster version for evaluating k
        if no_intercept:
            self.k = partial(no_intercept_polynomial_kernel, active_dims=active_dims)
        else:
            self.k = partial(polynomial_kernel, active_dims=active_dims)

    def default_params(self):
        return dict(
            degree=Parameter(
                value=2.0,
                trainable=False,
                bijector=Softplus(),
                prior=GammaPrior(),
            ),
            offset=Parameter(
                value=1.0,
                trainable=False,
                bijector=Identity(),
                prior=NormalPrior(loc=0.0, scale=1.0),
            ),
        )


class SquaredExponential(Kernel):
    """SquaredExponential kernel

    The Squared Exponential kernel:

        k(x, x') = exp( -∥ x - x'∥² / l)

    where l is the lengthscale.
    """

    def __init__(self, active_dims: ArrayLike = None) -> None:
        self._kernel_base = squared_exponential_kernel_base
        super().__init__(active_dims)
        # faster version for evaluating k
        self.k = partial(squared_exponential_kernel, active_dims=active_dims)

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                bijector=Softplus(),
                prior=GammaPrior(),
            )
        )


class Matern12(Kernel):
    """Matern12 kernel

    The Matern(ν=1/2) kernel:

        k(x, x') = exp(-z)

    where z = (d / l), l is the lengthscale, and d is the
    Euclidean distance ∥ x - x'∥.
    """

    def __init__(self, active_dims: ArrayLike = None) -> None:
        self._kernel_base = matern12_kernel_base
        super().__init__(active_dims)
        # faster version for evaluating k
        self.k = partial(matern12_kernel, active_dims=active_dims)

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                bijector=Softplus(),
                prior=GammaPrior(),
            )
        )


class Matern32(Kernel):
    """Matern32 kernel

    The Matern(ν=3/2) kernel:

        k(x, x') = (1 + √3 z) exp(-√3 z)

    where z = (d / l), l is the lengthscale, and d is the
    Euclidean distance ∥ x - x'∥.
    """

    def __init__(self, active_dims: ArrayLike = None) -> None:
        self._kernel_base = matern32_kernel_base
        super().__init__(active_dims)
        # faster version for evaluating k
        self.k = partial(matern32_kernel, active_dims=active_dims)

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                bijector=Softplus(),
                prior=GammaPrior(),
            )
        )


class Matern52(Kernel):
    """Matern52 kernel

    The Matern(ν=5/2) kernel:

        k(x, x') = (1 + √5 z + (5/3)z²) exp(-√5 z)

    where z = (d / l), l is the lengthscale, and d is the
    Euclidean distance ∥ x - x'∥.
    """

    def __init__(self, active_dims: ArrayLike = None) -> None:
        self._kernel_base = matern52_kernel_base
        super().__init__(active_dims)
        # faster version for evaluating k
        self.k = partial(matern52_kernel, active_dims=active_dims)

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                bijector=Softplus(),
                prior=GammaPrior(),
            )
        )


class ExpSinSquared(Kernel):
    """ExpSinSquared kernel

    The ExpSinSquared periodic kernel:

        k(x, x') = exp( -4 Σ_i sin²(π |x_i - x'_i|/ p) / l )

    where p is the periodicity and l is the lengthscale.

    This kernel is equivalent to a Squared Exponential kernel
    where the input is mapped as:

        x → [cos(x), sin(x)]

    """

    def __init__(self, active_dims=None):
        self._kernel_base = expsinsquared_kernel_base
        super().__init__(active_dims)

    def default_params(self):
        return dict(
            lengthscale=Parameter(
                value=1.0,
                trainable=True,
                bijector=Softplus(),
                prior=GammaPrior(),
            ),
            periodicity=Parameter(
                value=2 * jnp.pi,
                trainable=False,
                bijector=Softplus(),
                prior=GammaPrior(),
            ),
        )


class Sum(Kernel):
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
        # components
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        # kernel base
        # note that here and in the following we do not pass active_dims to the
        # sum_kernels etc because it is already set inside the two kernels
        self._kernel_base = sum_kernels(kernel1._kernel_base, kernel2._kernel_base)

        # inherit from Kernel class and pass active dims
        super().__init__(active_dims=None)

        # faster evaluation of the kernels
        self.k = sum_kernels(kernel1.k, kernel2.k)

        # derivative/hessian kernel
        self.d0k = sum_kernels(kernel1.d0k, kernel2.d0k)
        self.d1k = sum_kernels(kernel1.d1k, kernel2.d1k)
        self.d01k = sum_kernels(kernel1.d01k, kernel2.d01k)

        # derivative/hessian kernel-jacobian products
        self.d0kj = sum_kernels_jac(kernel1.d0kj, kernel2.d0kj)
        self.d1kj = sum_kernels_jac(kernel1.d1kj, kernel2.d1kj)
        self.d01kj = sum_kernels_jac2(kernel1.d01kj, kernel2.d01kj)

    def default_params(self):
        # simply delegate
        return {
            "kernel1": self.kernel1.default_params(),
            "kernel2": self.kernel2.default_params(),
        }


class Prod(Kernel):
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
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        # kernel base
        # note that here and in the following we do not pass active_dims to the
        # sum_kernels etc because it is already set inside the two kernels
        self._kernel_base = prod_kernels(kernel1._kernel_base, kernel2._kernel_base)

        # inherit from Kernel class and pass active dims
        super().__init__(active_dims=None)

        # faster evaluation of the kernels
        self.k = prod_kernels(kernel1.k, kernel2.k)

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

    def default_params(self):
        # simply delegate
        return {
            "kernel1": self.kernel1.default_params(),
            "kernel2": self.kernel2.default_params(),
        }
