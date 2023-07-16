import jax.numpy as jnp
from jax import random
from numpy.testing import assert_allclose

from gpx import kernels
from gpx.kernels.operations import kernel_center, kernel_center_test_test


def ref_linear_kernel(x, y=None):
    if y is None:
        return x @ x.T
    return x @ y.T


# ============================================================================
# Kernel centering
# ============================================================================


def test_centering_linear_kernel_train_train():
    """
    Checks that the kernel centering works for the simplest
    case (linear kernel). Here the centering is performed
    on k(x_train, x_train).
    """
    x = random.normal(random.PRNGKey(2023), shape=(10, 5))
    kernel = kernels.Linear()
    kernel_params = {}
    k_xx = kernel(x, x, kernel_params)

    # For a linear kernel, centering the kernel is equivalent
    # to evaluating the kernel on centered inputs
    k_xx_centered = kernel_center(k_xx, k_xx.mean(0))
    ref_k_xx_centered = ref_linear_kernel(x - x.mean(0))
    assert_allclose(k_xx_centered, ref_k_xx_centered)


def test_centering_linear_kernel_train_test():
    """
    Checks that the kernel centering works for the simplest
    case (linear kernel). Here the centering is performed
    on k(x_train, x_test).
    """
    x = random.normal(random.PRNGKey(2023), shape=(10, 5))
    y = random.normal(random.PRNGKey(2022), shape=(5, 5))
    kernel = kernels.Linear()
    kernel_params = {}
    k_xx = kernel(x, x, kernel_params)
    k_mean = k_xx.mean(0)
    k_xy = kernel(x, y, kernel_params)

    # Here we pass the mean over rows of K(train, train)
    k_xy_centered = kernel_center(k_xy, k_mean)
    ref_k_xy_centered = ref_linear_kernel(x - x.mean(0), y - x.mean(0))
    assert_allclose(k_xy_centered, ref_k_xy_centered)


def test_centering_linear_kernel_test_test():
    """
    Checks that the kernel centering works for the simplest
    case (linear kernel). Here the centering is performed
    on k(x_test, x_test).
    """
    x = random.normal(random.PRNGKey(2023), shape=(10, 5))
    y = random.normal(random.PRNGKey(2022), shape=(5, 5))
    kernel = kernels.Linear()
    kernel_params = {}
    k_xx = kernel(x, x, kernel_params)
    k_mean = k_xx.mean(0)
    k_xy = kernel(x, y, kernel_params)
    k_mean_train_test = k_xy.mean(0)
    k_yy = kernel(y, y, kernel_params)

    # Here we pass the mean over rows of K(train, train)
    # and the mean over rows of the cross kernel K(train, test)
    k_yy_centered = kernel_center_test_test(k_yy, k_mean, k_mean_train_test)
    ref_k_yy_centered = ref_linear_kernel(y - x.mean(0), y - x.mean(0))
    assert_allclose(k_yy_centered, ref_k_yy_centered)


def test_centering_poly2_train_train():
    """
    Checks that the kernel centering works for the case of
    a polynomial kernel of degree 2, for which the induced
    features are known exactly. Here the centering is performed
    on k(x_train, x_train).
    """
    x = random.normal(random.PRNGKey(2023), shape=(10, 2))
    kernel = kernels.Linear()
    kernel_params = {}

    # we take the dot product and square to
    # compute a polynomial kernel
    # k(x, y) = (x @ y)²
    k_xx = kernel(x, x, kernel_params) ** 2
    k_mean = k_xx.mean(0)

    # For this kernel it is easy to compute the
    # induced features. They are
    #   φ(x) = [x1x1, x1x2, x2x1, x2x2]
    phi = jnp.column_stack(
        [
            x[:, 0] * x[:, 0],
            x[:, 0] * x[:, 1],
            x[:, 1] * x[:, 0],
            x[:, 1] * x[:, 1],
        ]
    )

    # check for equality of the non-centered kernel
    ref_k_xx = ref_linear_kernel(phi, phi)
    assert_allclose(k_xx, ref_k_xx)

    # check for equality of the centered kernel
    k_xx_centered = kernel_center(k_xx, k_mean)
    ref_k_xx_centered = ref_linear_kernel(phi - phi.mean(0), phi - phi.mean(0))
    assert_allclose(k_xx_centered, ref_k_xx_centered)
