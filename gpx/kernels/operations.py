from typing import Callable

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
        return kernel_func1(x1, x2, params1).repeat(3, axis=axis) * deriv_func2(
            x1, x2, params2, jacobian
        ) + deriv_func1(x1, x2, params1, jacobian) * kernel_func2(
            x1, x2, params2
        ).repeat(
            3, axis=axis
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
        return (
            kernel_func1(x1, x2, params1).repeat(3, axis=0).repeat(3, axis=-1)
            * deriv01_func2(x1, x2, params2, jacobian1, jacobian2)
            + deriv0_func1(x1, x2, params1, jacobian1).repeat(3, axis=-1)
            * deriv1_func2(x1, x2, params2, jacobian2).repeat(3, axis=0)
            + deriv1_func1(x1, x2, params1, jacobian2).repeat(3, axis=0)
            * deriv0_func2(x1, x2, params2, jacobian1).repeat(3, axis=-1)
            + deriv01_func1(x1, x2, params1, jacobian1, jacobian2)
            * kernel_func2(x1, x2, params2).repeat(3, axis=0).repeat(3, axis=-1)
        )

    return kernel
