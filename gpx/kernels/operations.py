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
