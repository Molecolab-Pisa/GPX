from typing import Dict

import jax.numpy as jnp
from jax import jit

from .kernelizers import kernelize
from ..utils import squared_distances
from ..parameters.parameter import Parameter


# =============================================================================
# Squared Exponential Kernel
# =============================================================================


def _squared_exponential_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, lengthscale: float
) -> jnp.ndarray:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    return jnp.exp(-jnp.sum((z1 - z2) ** 2))


@jit
def squared_exponential_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, params: Dict[str, Parameter]
) -> jnp.ndarray:
    lengthscale = params["lengthscale"].value
    return _squared_exponential_kernel_base(x1, x2, lengthscale)


def _squared_exponential_kernel(
    x1: jnp.ndarray, x2: jnp.ndarray, lengthscale: float
) -> jnp.ndarray:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d2 = squared_distances(z1, z2)
    return jnp.exp(-d2)


@jit
def squared_exponential_kernel(
    x1: jnp.ndarray, x2: jnp.ndarray, params: Dict[str, Parameter]
) -> jnp.ndarray:
    lengthscale = params["lengthscale"].value
    return _squared_exponential_kernel(x1, x2, lengthscale)


# =============================================================================
# Matern(1/2) Kernel
# =============================================================================


def _matern12_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, lengthscale: float
) -> jnp.ndarray:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d = jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return jnp.exp(-d)


@jit
def matern12_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, params: Dict[str, Parameter]
) -> jnp.ndarray:
    lengthscale = params["lengthscale"].value
    return _matern12_kernel_base(x1, x2, lengthscale)


matern12_kernel = kernelize(matern12_kernel_base)


# =============================================================================
# Matern(3/2) Kernel
# =============================================================================


def _matern32_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, lengthscale: float
) -> jnp.ndarray:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d = jnp.sqrt(3.0) * jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return (1.0 + d) * jnp.exp(-d)


@jit
def matern32_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, params: Dict[str, Parameter]
) -> jnp.ndarray:
    lengthscale = params["lengthscale"]
    return _matern32_kernel_base(x1, x2, lengthscale)


matern32_kernel = kernelize(matern32_kernel_base)


# =============================================================================
# Matern(3/2) Kernel
# =============================================================================


def _matern52_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, lengthscale: float
) -> jnp.ndarray:
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    d = jnp.sqrt(5.0) * jnp.sqrt(jnp.sum((z1 - z2) ** 2))
    return (1.0 + d + d**2 / 3.0) * jnp.exp(-d)


@jit
def matern52_kernel_base(
    x1: jnp.ndarray, x2: jnp.ndarray, params: Dict[str, Parameter]
) -> jnp.ndarray:
    lengthscale = params["lengthscale"]
    return _matern52_kernel_base(x1, x2, lengthscale)


matern52_kernel = kernelize(matern52_kernel_base)


# =============================================================================
# Kernel aliases
# =============================================================================

se_kernel = squared_exponential_kernel
rbf_kernel = squared_exponential_kernel
m12_kernel = matern12_kernel
m32_kernel = matern32_kernel
m52_kernel = matern52_kernel
