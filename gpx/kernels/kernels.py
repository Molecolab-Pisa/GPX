from typing import Callable, Dict

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
# Constant Kernel
# =============================================================================


def _constant_kernel_base(x1: ArrayLike, x2: ArrayLike, variance: ArrayLike) -> Array:
    return variance


@jit
def constant_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    variance = params["variance"].value
    return _constant_kernel_base(x1, x2, variance)


def _constant_kernel(x1: ArrayLike, x2: ArrayLike, variance: ArrayLike) -> Array:
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
# Polynomial Kernel
# =============================================================================


def _polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return (offset + jnp.dot(x1, x2)) ** degree


@jit
def polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    return _polynomial_kernel_base(x1, x2, degree, offset)


def _polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return (offset + x1 @ x2.T) ** degree


@jit
def polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    return _polynomial_kernel(x1, x2, degree, offset)


# =============================================================================
# No Intercept Polynomial Kernel
# =============================================================================


def _no_intercept_polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return ((offset + jnp.dot(x1, x2)) ** degree) - (offset**degree)


@jit
def no_intercept_polynomial_kernel_base(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    return _no_intercept_polynomial_kernel_base(x1, x2, degree, offset)


def _no_intercept_polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, degree: ArrayLike, offset: ArrayLike
) -> Array:
    return ((offset + x1 @ x2.T) ** degree) - (offset**degree)


@jit
def no_intercept_polynomial_kernel(
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    degree = params["degree"].value
    offset = params["offset"].value
    return _no_intercept_polynomial_kernel(x1, x2, degree, offset)


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
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _squared_exponential_kernel_base(x1, x2, lengthscale)


def _squared_exponential_kernel(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
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
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern12_kernel_base(x1, x2, lengthscale)


def _matern12_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike) -> Array:
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
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern32_kernel_base(x1, x2, lengthscale)


def _matern32_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike) -> Array:
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
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    return _matern52_kernel_base(x1, x2, lengthscale)


def _matern52_kernel(x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike) -> Array:
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
    x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]
) -> Array:
    lengthscale = params["lengthscale"].value
    period = params["periodicity"].value
    return _expsinsquared_kernel_base(x1, x2, lengthscale, period)


# =============================================================================
# Filter Active Dimensions
# =============================================================================


def active_dims_filter(kernel_func: Callable, active_dims: ArrayLike) -> Callable:
    """filters the active dimension in the input

    Given a kernel function operating on two samples x1 and x2, this function
    yields another kernel function operating on filtered x1 and x2.
    The filtered input retains only the columns specified in `active_dims`, e.g.,
    if `active_dims = [0, 1]`, only the first two columns of x1 are retained.

    Args:
        kernel_func: kernel function
        active_dims: active dimensions (columns) to retain in the input
    Returns:
        kernel: kernel function operating on filtered inputs
    """

    def kernel(x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]):
        x1 = x1[:, active_dims]
        x2 = x2[:, active_dims]
        return kernel_func(x1, x2, params)

    return kernel


def active_dims_filter_jac(kernel_func: Callable, active_dims: ArrayLike) -> Callable:
    """filters the active dimension in the input

    Given a kernel function operating on two samples x1 and x2, this function
    yields another kernel function operating on filtered x1 and x2 and jacobian.
    The filtered input retains only the columns specified in `active_dims`, e.g.,
    if `active_dims = [0, 1]`, only the first two columns of x1 are retained.

    Args:
        kernel_func: kernel function
        active_dims: active dimensions (columns) to retain in the input
    Returns:
        kernel: kernel function operating on filtered inputs
    """

    def kernel(
        x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], jacobian: ArrayLike
    ):
        x1 = x1[:, active_dims]
        x2 = x2[:, active_dims]
        jacobian = jacobian[:, active_dims, :]
        return kernel_func(x1, x2, params, jacobian)

    return kernel


def active_dims_filter_jac2(kernel_func: Callable, active_dims: ArrayLike) -> Callable:
    """filters the active dimension in the input

    Given a kernel function operating on two samples x1 and x2, this function
    yields another kernel function operating on filtered x1 and x2, jacobian1
    and jacobian2. The filtered input retains only the columns specified in
    `active_dims`, e.g., if `active_dims = [0, 1]`, only the first two columns
    of x1 are retained.

    Args:
        kernel_func: kernel function
        active_dims: active dimensions (columns) to retain in the input
    Returns:
        kernel: kernel function operating on filtered inputs
    """

    def kernel(
        x1: ArrayLike,
        x2: ArrayLike,
        params: Dict[str, Parameter],
        jacobian1: ArrayLike,
        jacobian2: ArrayLike,
    ):
        x1 = x1[:, active_dims]
        x2 = x2[:, active_dims]
        jacobian1 = jacobian1[:, active_dims, :]
        jacobian2 = jacobian2[:, active_dims, :]
        return kernel_func(x1, x2, params, jacobian1, jacobian2)

    return kernel


def identity_filter(kernel_func: Callable, active_dims: ArrayLike) -> Callable:
    """filters every dimension in the input

    Filter with no effects. The `active_dims` argument is not used.

    Args:
        kernel_func: kernel function
        active_dims: active dimensions (columns) to retain in the input
    Returns:
        kernel_func: kernel function
    """

    def kernel(x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter]):
        return kernel_func(x1, x2, params)

    return kernel


def identity_filter_jac(kernel_func: Callable, active_dims: ArrayLike) -> Callable:
    """filters every dimension in the input

    Given a kernel function operating on two samples x1 and x2, this function
    yields another kernel function operating on filtered x1 and x2, jacobian1
    and jacobian2. This filter has no effects. The `active_dims` argument is not used.

    Args:
        kernel_func: kernel function
        active_dims: active dimensions (columns) to retain in the input
    Returns:
        kernel: kernel function operating on filtered inputs
    """

    def kernel(
        x1: ArrayLike, x2: ArrayLike, params: Dict[str, Parameter], jacobian: ArrayLike
    ):
        return kernel_func(x1, x2, params, jacobian)

    return kernel


def identity_filter_jac2(kernel_func: Callable, active_dims: ArrayLike) -> Callable:
    """filters every dimension in the input

    Given a kernel function operating on two samples x1 and x2, this function
    yields another kernel function operating on filtered x1 and x2, jacobian1
    and jacobian2. This filter has no effects. The `active_dims` argument is not used.

    Args:
        kernel_func: kernel function
        active_dims: active dimensions (columns) to retain in the input
    Returns:
        kernel: kernel function operating on filtered inputs
    """

    def kernel(
        x1: ArrayLike,
        x2: ArrayLike,
        params: Dict[str, Parameter],
        jacobian1: ArrayLike,
        jacobian2: ArrayLike,
    ):
        return kernel_func(x1, x2, params, jacobian1, jacobian2)

    return kernel


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
        self.k = self.filter_input(kernelize(self._kernel_base), self.active_dims)

        # derivative/hessian kernel
        self.d0k = self.filter_input(
            grad_kernelize(argnums=0, with_jacob=False)(self._kernel_base),
            self.active_dims,
        )
        self.d1k = self.filter_input(
            grad_kernelize(argnums=1, with_jacob=False)(self._kernel_base),
            self.active_dims,
        )
        self.d01k = self.filter_input(
            grad_kernelize(argnums=(0, 1), with_jacob=False)(self._kernel_base),
            self.active_dims,
        )

        # derivative/hessian kernel-jacobian products
        self.d0kj = self.filter_input_jac(
            grad_kernelize(argnums=0, with_jacob=True)(self._kernel_base),
            self.active_dims,
        )
        self.d1kj = self.filter_input_jac(
            grad_kernelize(argnums=1, with_jacob=True)(self._kernel_base),
            self.active_dims,
        )
        self.d01kj = self.filter_input_jac2(
            grad_kernelize(argnums=(0, 1), with_jacob=True)(self._kernel_base),
            self.active_dims,
        )

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        if value is None:
            self._active_dims = value
            self.filter_input = identity_filter
            self.filter_input_jac = identity_filter_jac
            self.filter_input_jac2 = identity_filter_jac2
        else:
            self._active_dims = value
            self.filter_input = active_dims_filter
            self.filter_input_jac = active_dims_filter_jac
            self.filter_input_jac2 = active_dims_filter_jac2

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
        self.k = self.filter_input(constant_kernel, self.active_dims)

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
        self.k = self.filter_input(linear_kernel, self.active_dims)

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
            self.k = self.filter_input(no_intercept_polynomial_kernel, self.active_dims)
        else:
            self.k = self.filter_input(polynomial_kernel, self.active_dims)

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
        self.k = self.filter_input(squared_exponential_kernel, self.active_dims)

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
        self.k = self.filter_input(matern12_kernel, self.active_dims)

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
        self.k = self.filter_input(matern32_kernel, self.active_dims)

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
        self.k = self.filter_input(matern52_kernel, self.active_dims)

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

    def __init__(
        self, kernel1: Kernel, kernel2: Kernel, active_dims: ArrayLike = None
    ) -> None:
        # components
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        # kernel base
        self._kernel_base = sum_kernels(kernel1._kernel_base, kernel2._kernel_base)

        # inherit from Kernel class and pass active dims
        super().__init__(active_dims)

        # faster evaluation of the kernels
        self.k = self.filter_input(sum_kernels(kernel1.k, kernel2.k), self.active_dims)

        # derivative/hessian kernel
        self.d0k = self.filter_input(
            sum_kernels(kernel1.d0k, kernel2.d0k), self.active_dims
        )
        self.d1k = self.filter_input(
            sum_kernels(kernel1.d1k, kernel2.d1k), self.active_dims
        )
        self.d01k = self.filter_input(
            sum_kernels(kernel1.d01k, kernel2.d01k), self.active_dims
        )

        # derivative/hessian kernel-jacobian products
        self.d0kj = self.filter_input_jac(
            sum_kernels_jac(kernel1.d0kj, kernel2.d0kj), self.active_dims
        )
        self.d1kj = self.filter_input_jac(
            sum_kernels_jac(kernel1.d1kj, kernel2.d1kj), self.active_dims
        )
        self.d01kj = self.filter_input_jac2(
            sum_kernels_jac2(kernel1.d01kj, kernel2.d01kj), self.active_dims
        )

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

    def __init__(
        self, kernel1: Kernel, kernel2: Kernel, active_dims: ArrayLike = None
    ) -> None:
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        # kernel base
        self._kernel_base = prod_kernels(kernel1._kernel_base, kernel2._kernel_base)

        # inherit from Kernel class and pass active dims
        super().__init__(active_dims)

        # faster evaluation of the kernels
        self.k = self.filter_input(prod_kernels(kernel1.k, kernel2.k), self.active_dims)

        # derivative/hessian kernel
        self.d0k = self.filter_input(
            prod_kernels_deriv(kernel1.k, kernel2.k, kernel1.d0k, kernel2.d0k, 0),
            self.active_dims,
        )
        self.d1k = self.filter_input(
            prod_kernels_deriv(kernel1.k, kernel2.k, kernel1.d1k, kernel2.d1k, -1),
            self.active_dims,
        )
        self.d01k = self.filter_input(
            prod_kernels_deriv01(
                kernel1.k,
                kernel2.k,
                kernel1.d0k,
                kernel2.d0k,
                kernel1.d1k,
                kernel2.d1k,
                kernel1.d01k,
                kernel2.d01k,
            ),
            self.active_dims,
        )

        # derivative/hessian kernel-jacobian products
        self.d0kj = self.filter_input_jac(
            prod_kernels_deriv_jac(kernel1.k, kernel2.k, kernel1.d0kj, kernel2.d0kj, 0),
            self.active_dims,
        )
        self.d1kj = self.filter_input_jac(
            prod_kernels_deriv_jac(
                kernel1.k, kernel2.k, kernel1.d1kj, kernel2.d1kj, -1
            ),
            self.active_dims,
        )
        self.d01kj = self.filter_input_jac2(
            prod_kernels_deriv01_jac(
                kernel1.k,
                kernel2.k,
                kernel1.d0kj,
                kernel2.d0kj,
                kernel1.d1kj,
                kernel2.d1kj,
                kernel1.d01kj,
                kernel2.d01kj,
            ),
            self.active_dims,
        )

    def default_params(self):
        # simply delegate
        return {
            "kernel1": self.kernel1.default_params(),
            "kernel2": self.kernel2.default_params(),
        }
