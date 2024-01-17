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
# Kernel Decorator
# =============================================================================


def sum_kernels(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    def kernel(x1, x2, params):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1) + kernel_func2(x1, x2, params2)

    return kernel


def sum_kernels_jac(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    def kernel(x1, x2, params, jacobian):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1, jacobian) + kernel_func2(
            x1, x2, params2, jacobian
        )

    return kernel


def sum_kernels_jac2(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    def kernel(x1, x2, params, jacobian1, jacobian2):
        params1 = params["kernel1"]
        params2 = params["kernel2"]
        return kernel_func1(x1, x2, params1, jacobian1, jacobian2) + kernel_func2(
            x1, x2, params2, jacobian1, jacobian2
        )

    return kernel


def prod_kernels(kernel_func1: Callable, kernel_func2: Callable) -> Callable:
    def kernel(x1, x2, params):
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
    def kernel(x1, x2, params):
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
    def kernel(x1, x2, params):
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
    def kernel(x1, x2, params, jacobian):
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
    def kernel(x1, x2, params, jacobian1, jacobian2):
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
