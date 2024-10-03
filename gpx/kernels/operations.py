# GPX: gaussian process regression in JAX
# Copyright (C) 2023  GPX authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

from typing import Callable

from jax import Array
from jax.typing import ArrayLike

# =============================================================================
# Kernel Centering
# =============================================================================


def kernel_center(k: ArrayLike, k_mean: ArrayLike) -> Array:
    """Center the kernel

    Center the kernel, which is equivalent to center the input
    in the space induced by the kernel, e.g.:

        K_centered(x_i, x_j) = (φ(x_i) - φ_mean)^T (φ(x_j) - φ_mean)

    The centering can be computed without knowing φ(x) and φ_mean
    explicitely:

        K_centered(x_i, x_j) = K(x_i, x_j) - (1/N) Σ_l x_l x_j -
                               - (1/N) Σ_m x_i x_m +
                               + (1/N²) Σ_lm x_l x_m

    where N is the number of training points. This means that we can
    compute it by only knowing the mean over rows or columns of
    K.

    The present function requires the mean over rows of K, and it can
    be applied also to transform a kernel matrix evaluated between
    training points and test ponts (for which you don't want to compute
    the mean). In this case, the provided kernel matrix must be evaluated
    as k(x_train, x_test), *not* as k(x_test, x_train).

    Args:
        k: kernel matrix, can be evaluated only on x_train,
           k(x_train, x_train), or can be evaluated between
           x_train and x_test as k(x_train, x_test)
        k_mean: mean over the rows of k(x_train, x_train)
                (e.g., jnp.mean(k, axis=0))
    Returns:
        k_centered: centered kernel
    """
    return k - k.mean(0)[None, :] - k_mean[:, None] + k_mean.mean()


def kernel_center_test_test(
    k: Array, k_mean_train: ArrayLike, k_mean_train_test
) -> Array:
    """Center the kernel using the training mean

    Center the kernel, which is equivalent to center the input
    in the space induced by the kernel, e.g.:

        K_centered(x_i, x_j) = (φ(x_i) - φ_mean)^T (φ(x_j) - φ_mean)

    Here the centering is performed using the mean over training
    points. This function should be used to center the kernel
    evaluated between test points, i.e. k(x_test, x_test)

    Args:
        k: kernel matrix, can be evaluated only on x_train,
           k(x_train, x_train), or can be evaluated between
           x_train and x_test as k(x_train, x_test)
        k_mean_train: mean over the rows of k(x_train, x_train)
                (e.g., jnp.mean(k, axis=0))
        k_mean_train_test: mean over the rows of k(x_train, x_test)
    Returns:
        k_centered: centered kernel
    """
    return (
        k
        - k_mean_train_test[None, :]
        - k_mean_train_test[:, None]
        + k_mean_train.mean()
    )


# =============================================================================
# Sum of two kernels
# =============================================================================
#


def sum_kernels(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    """defines a kernel as the sum of two kernels

    Defines a new kernel function as the sum of two kernel functions.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    # note: the kernels must have the same dimensions because it's just
    # (n_samples1, n_samples2) if kernel_func* computes the kernel
    # (n_samples1*n_features, n_samples2) if kernel_func* computes the d0
    # (n_samples1, n_samples2*n_features) if kernel_func* computes the d1
    # (n_samples1*n_features, n_samples2*n_features) if [...] d01
    def kernel(x1, x2, params, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1) + kernel_func2(x1, x2, params2)

    return kernel


def sum_kernels_jac(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    """defines a kernel as the sum of two kernels

    Defines a new kernel function as the sum of two kernel functions.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.
    The new kernel accepts a single jacobian.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    # here the two functions compute the product with one jacobian, so
    # their shape is consistent
    def kernel(x1, x2, params, jacobian, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1, jacobian) + kernel_func2(
            x1, x2, params2, jacobian
        )

    return kernel


def sum_kernels_jac2(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    """defines a kernel as the sum of two kernels

    Defines a new kernel function as the sum of two kernel functions.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.
    The new kernel accepts one jacobian for x1 and one for x2.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    # here both functions compute the hessian with two jacobians, so their
    # shape is consistent
    def kernel(x1, x2, params, jacobian1, jacobian2, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1, jacobian1, jacobian2) + kernel_func2(
            x1, x2, params2, jacobian1, jacobian2
        )

    return kernel


def prod_kernels(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    """defines a kernel as the product of two kernels

    Defines a new kernel function as the product of two kernel functions.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.
    The new kernel accepts one jacobian for x1 and one for x2.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    # consistent shapes, see "sum_kernels"
    def kernel(x1, x2, params, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1) * kernel_func2(x1, x2, params2)

    return kernel


def prod_kernels_deriv(
    kernel_func1: Callable,
    kernel_func2: Callable,
    deriv_func1: Callable,
    deriv_func2: Callable,
    axis: int,
) -> Callable:
    """defines a kernel as the derivative of the product of two kernels

    Defines a new kernel function as the derivative of the product of two
    kernel functions.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.
    The new kernel accepts one jacobian for x1 and one for x2.
    If the derivative is w.r.t. the first argument:

        k = (∂₀k₁)k₂ + k₁(∂₀k₂)

    Otherwise change ∂₀ with ∂₁.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    # here we must ensure that the shapes are consistent by repeating the entries
    # of the kernels that are not derived. For each sample, the jacobian kernel
    # has a shape multiplied by n_features (M below). Each of these entries multiplies
    # the entry of that sample in the non-derived kernel.
    def kernel(x1, x2, params, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        _, M = x1.shape
        return kernel_func1(x1, x2, params1).repeat(M, axis=axis) * deriv_func2(
            x1, x2, params2
        ) + deriv_func1(x1, x2, params1) * kernel_func2(x1, x2, params2).repeat(
            M, axis=axis
        )

    return kernel


def prod_kernels_deriv01(
    kernel_func1: Callable,
    kernel_func2: Callable,
    deriv0_func1: Callable,
    deriv0_func2: Callable,
    deriv1_func1: Callable,
    deriv1_func2: Callable,
    deriv01_func1: Callable,
    deriv01_func2: Callable,
) -> Callable:
    """defines a kernel as the hessian of the product of two kernels

    Defines a new kernel function as the hessian of the product of two
    kernel functions.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.
    The new kernel accepts one jacobian for x1 and one for x2.

        k = (∂₀∂₁k₁)k₂ + (∂₁k₁)(∂₀k₂) + (∂₀k₁)(∂₁k₂) + k₁(∂₀∂₁k₂)


    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    def kernel(x1, x2, params, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        _, M = x2.shape
        return (
            kernel_func1(x1, x2, params1).repeat(M, axis=0).repeat(M, axis=-1)
            * deriv01_func2(x1, x2, params2)
            + deriv0_func1(x1, x2, params1).repeat(M, axis=-1)
            * deriv1_func2(x1, x2, params2).repeat(M, axis=0)
            + deriv1_func1(x1, x2, params1).repeat(M, axis=0)
            * deriv0_func2(x1, x2, params2).repeat(M, axis=-1)
            + deriv01_func1(x1, x2, params1)
            * kernel_func2(x1, x2, params2).repeat(M, axis=0).repeat(M, axis=-1)
        )

    return kernel


def prod_kernels_deriv_jac(
    kernel_func1: Callable,
    kernel_func2: Callable,
    deriv_func1: Callable,
    deriv_func2: Callable,
    axis: int,
) -> Callable:
    """defines a kernel as the derivative of the product of two kernels

    Defines a new kernel function as the derivative of the product of two
    kernel functions.
    The function also computes the product with the Jacobian of the input.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.
    The new kernel accepts one jacobian for x1 and one for x2.
    If the derivative is w.r.t. the first argument:

        k = (J∂₀k₁)k₂ + k₁(J∂₀k₂)

    Otherwise change ∂₀ with ∂₁.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    def kernel(x1, x2, params, jacobian, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        _, _, jv = jacobian.shape
        return kernel_func1(x1, x2, params1).repeat(jv, axis=axis) * deriv_func2(
            x1, x2, params2, jacobian
        ) + deriv_func1(x1, x2, params1, jacobian) * kernel_func2(
            x1, x2, params2
        ).repeat(
            jv, axis=axis
        )

    return kernel


def prod_kernels_deriv0_jaccoef(
    kernel_func1: Callable,
    kernel_func2: Callable,
    deriv_func1: Callable,
    deriv_func2: Callable,
) -> Callable:
    """defines a kernel as the derivative of the product of two kernels

    Defines a new kernel function as the derivative of the product of two
    kernel functions.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    def kernel(x1, x2, params, jaccoef, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1) * deriv_func2(
            x1, x2, params2, jaccoef
        ) + deriv_func1(x1, x2, params1, jaccoef) * kernel_func2(x1, x2, params2)

    return kernel


def prod_kernels_deriv01_jac(
    kernel_func1: Callable,
    kernel_func2: Callable,
    deriv0_func1: Callable,
    deriv0_func2: Callable,
    deriv1_func1: Callable,
    deriv1_func2: Callable,
    deriv01_func1: Callable,
    deriv01_func2: Callable,
) -> Callable:
    """defines a kernel as the hessian of the product of two kernels

    Defines a new kernel function as the hessian of the product of two
    kernel functions.
    The function also computes the product with the Jacobian of the inputs.
    The new kernel will accept a "params" argument storing the parameters
    for the first and second kernel as "kernel1" and "kernel2", respectively.
    The new kernel accepts one jacobian for x1 and one for x2.
    If the derivative is w.r.t. the first argument:

        k = (J₀∂₀∂₁k₁J₁)k₂ + (∂₁k₁J₁)(J₀∂₀k₂) + (J₀∂₀k₁)(∂₁k₂J₁) + k₁(J₀∂₀∂₁k₂J₁)

    Otherwise change ∂₀ with ∂₁.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    def kernel(x1, x2, params, jacobian1, jacobian2, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        _, _, jv1 = jacobian1.shape
        _, _, jv2 = jacobian2.shape
        return (
            kernel_func1(x1, x2, params1).repeat(jv1, axis=0).repeat(jv2, axis=-1)
            * deriv01_func2(x1, x2, params2, jacobian1, jacobian2)
            + deriv0_func1(x1, x2, params1, jacobian1).repeat(jv2, axis=-1)
            * deriv1_func2(x1, x2, params2, jacobian2).repeat(jv1, axis=0)
            + deriv1_func1(x1, x2, params1, jacobian2).repeat(jv1, axis=0)
            * deriv0_func2(x1, x2, params2, jacobian1).repeat(jv2, axis=-1)
            + deriv01_func1(x1, x2, params1, jacobian1, jacobian2)
            * kernel_func2(x1, x2, params2).repeat(jv1, axis=0).repeat(jv2, axis=-1)
        )

    return kernel


def prod_kernels_deriv01_jaccoef(
    kernel_func1: Callable,
    kernel_func2: Callable,
    deriv0_func1: Callable,
    deriv0_func2: Callable,
    deriv1_func1: Callable,
    deriv1_func2: Callable,
    deriv01_func1: Callable,
    deriv01_func2: Callable,
) -> Callable:
    """defines a kernel as the hessian of the product of two kernels

    Defines a new kernel function as the hessian of the product of two
    kernel functions.

    Note: active_dims is there as a placeholder, and it is not passed to
          the two kernels.
    """

    def kernel(x1, x2, params, jaccoef, jacobian, active_dims=None):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        _, _, jv2 = jacobian.shape
        return (
            kernel_func1(x1, x2, params1).repeat(jv2, axis=-1)
            * deriv01_func2(x1, x2, params2, jaccoef, jacobian)
            + deriv0_func1(x1, x2, params1, jaccoef).repeat(jv2, axis=-1)
            * deriv1_func2(x1, x2, params2, jacobian)
            + deriv1_func1(x1, x2, params1, jacobian)
            * deriv0_func2(x1, x2, params2, jaccoef).repeat(jv2, axis=-1)
            + deriv01_func1(x1, x2, params1, jaccoef, jacobian)
            * kernel_func2(x1, x2, params2).repeat(jv2, axis=-1)
        )

    return kernel
