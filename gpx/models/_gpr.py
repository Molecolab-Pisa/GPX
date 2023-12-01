from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit
from jax.scipy.sparse.linalg import cg
from jax.typing import ArrayLike

from ..lanczos import lanczos_logdet
from ..parameters import Parameter

ParameterDict = Dict[str, Parameter]
Kernel = Any

# Functions to compute the kernel matrices needed for GPR


@partial(jit, static_argnums=(3, 4))
def _A_lhs(
    x1: ArrayLike,
    x2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
) -> Array:
    """lhs of A x = b

    Builds the left hand side (lhs) of A x = b for GPR.
    Dense implementation: A is built all at once.

        A = K(x, x) + σ²I
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _ = x1.shape
    C_mm = kernel(x1, x2, kernel_params)

    if noise:
        C_mm = C_mm + (sigma**2 + 1e-10) * jnp.eye(m)

    return C_mm


@partial(jit, static_argnums=(5, 6))
def _A_derivs_lhs(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
) -> Array:
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    C_mm = kernel.d01kj(x1, x2, kernel_params, jacobian1, jacobian2)

    if noise:
        C_mm = C_mm + (sigma**2 + 1e-10) * jnp.eye(C_mm.shape[0])

    return C_mm


def _Ax_lhs_fun(
    x1: ArrayLike,
    x2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
) -> Callable[ArrayLike, Array]:
    """matrix-vector function for the lhs of A x = b

    Builds a function that computes the matrix-vector
    product of the left hand side (lhs) of A x = b.
    The matrix-vector product is computed iteratively.

        A = K(x, x) + σ²I
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    if noise:

        @jit
        def matvec(z):
            def update_row(carry, x1s):
                # using 'kernel' adds a singleton dimension at 0
                # that must be discarded
                kernel_row = kernel(x1s[jnp.newaxis], x2, kernel_params).squeeze(axis=0)
                kernel_row = kernel_row.at[carry].add(sigma**2 + 1e-10)
                rowvec = jnp.dot(kernel_row, z)
                carry = carry + 1
                return carry, rowvec

            _, res = jax.lax.scan(update_row, 0, x1)
            return res

    else:

        @jit
        def matvec(z):
            def update_row(carry, x1s):
                kernel_row = kernel(x1s[jnp.newaxis], x2, kernel_params).squeeze(axis=0)
                rowvec = jnp.dot(kernel_row, z)
                return carry, rowvec

            _, res = jax.lax.scan(update_row, 0, x1)
            return res

    return matvec


def _Ax_derivs_lhs_fun(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
) -> Callable[ArrayLike, Array]:
    """matrix-vector function for the lhs of A x = b

    Builds a function that computes the matrix-vector
    product of the left hand side (lhs) of A x = b when
    training on derivative values only.
    The matrix-vector product is computed iteratively.

        A = K(x, x) + σ²I
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    nr = jacobian1.shape[2]

    if noise:

        @jit
        def matvec(z):
            def update_row(carry, xj):
                x1s, j1s = xj
                # this is a stripe, not a row: the first dim is the
                # number of derivatives
                kernel_row = kernel.d01kj(
                    x1s[jnp.newaxis], x2, kernel_params, j1s[jnp.newaxis], jacobian2
                )
                # we have to add the noise + jitter to the diagonal
                # we do so by updating the stripe block by block, where
                # the block has the dimension of the number of rows in the
                # slice
                start_indices = (0, nr * carry)
                jnoise = (sigma**2 + 1e-10) * jnp.eye(nr)
                fill = (
                    jax.lax.dynamic_slice(kernel_row, start_indices, (nr, nr)) + jnoise
                )
                kernel_row = jax.lax.dynamic_update_slice(
                    kernel_row,
                    fill,
                    start_indices,
                )
                rowvec = jnp.dot(kernel_row, z)
                carry = carry + 1
                return carry, rowvec

            _, res = jax.lax.scan(update_row, 0, (x1, jacobian1))
            res = jnp.concatenate(res, axis=0)
            return res

    else:

        @jit
        def matvec(z):
            def update_row(carry, xj):
                x1s, j1s = xj
                # this is a stripe, not a row: the first dim is the
                # number of derivatives
                kernel_row = kernel.d01kj(
                    x1s[jnp.newaxis], x2, kernel_params, j1s[jnp.newaxis], jacobian2
                )
                rowvec = jnp.dot(kernel_row, z)
                return carry, rowvec

            _, res = jax.lax.scan(update_row, 0, (x1, jacobian1))
            res = jnp.concatenate(res, axis=0)
            return res

    return matvec


# Functions to fit a GPR


@partial(jit, static_argnums=(3, 4))
def _fit_dense(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    """fits a standard GPR with Cholesky

    Fits a standard GPR using the Cholesky decomposition
    to solve the linear system

    μ = m(y)
    c = (K(x, x) + σ²I)⁻¹y
    """
    mu = mean_function(y)
    y = y - mu
    C_mm = _A_lhs(x1=x, x2=x, params=params, kernel=kernel, noise=True)
    c = jnp.linalg.solve(C_mm, y)
    return c, mu


@partial(jit, static_argnums=(3, 4))
def _fit_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    """fits a standard GPR iteratively

    Fits a standard GPR solving the linear system iteratively
    with Conjugated Gradient.

    μ = m(y)
    c = (K(x, x) + σ²I)⁻¹y
    """
    mu = mean_function(y)
    y = y - mu
    matvec = _Ax_lhs_fun(x1=x, x2=x, params=params, kernel=kernel, noise=True)
    c, _ = cg(matvec, y)
    return c, mu


@partial(jit, static_argnums=(4, 5))
def _fit_derivs_dense(
    params: ParameterDict,
    x: ArrayLike,
    jacobian: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    """fits a standard GPR with Cholesky when training on derivatives

    Fits a standard GPR using the Cholesky decomposition to solve
    the linear system when training on derivative values.

    μ = m(y)
    c = (K(x, x) + σ²I)⁻¹y
    """
    kernel = partial(kernel.d01kj, jacobian1=jacobian, jacobian2=jacobian)
    # also flatten y
    y = y.reshape(-1, 1)
    return _fit_dense(params, x, y, kernel, mean_function)


@partial(jit, static_argnums=(4, 5))
def _fit_derivs_iter(
    params: ParameterDict,
    x: ArrayLike,
    jacobian: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    """fits a standard GPR iteratively when training on derivatives

    Fits a standard GPR solving the linear system iteratively with
    Conjugate Gradient when training on derivative values.

    μ = m(y)
    c = (K(x, x) + σ²I)⁻¹y
    """
    # flatten y
    y = y.reshape(-1, 1)
    mu = mean_function(y)
    y = y - mu
    matvec = _Ax_derivs_lhs_fun(
        x1=x,
        jacobian1=jacobian,
        x2=x,
        jacobian2=jacobian,
        params=params,
        kernel=kernel,
        noise=True,
    )
    c, _ = cg(matvec, y)
    return c, mu


# functions to predict with a GPR


@partial(jit, static_argnums=(5, 6))
def _predict_dense(
    params: ParameterDict,
    x_train: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
    full_covariance: Optional[bool] = False,
) -> Union[Array, Tuple[Array, Array]]:
    """predicts with GPR

    Predicts with GPR, by first building the full kernel matrix
    and then contracting with the linear coefficients.

    μ = K_nm (K_mm + σ²)⁻¹y
    C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn
    """
    K_mn = _A_lhs(x1=x_train, x2=x, params=params, kernel=kernel, noise=False)
    mu = mu + jnp.dot(K_mn.T, c)

    if full_covariance:
        C_mm = _A_lhs(x1=x_train, x2=x_train, params=params, kernel=kernel, noise=True)
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = _A_lhs(x1=x, x2=x, params=params, kernel=kernel, noise=False)
        C_nn = C_nn - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn

    return mu


@partial(jit, static_argnums=(5,))
def _predict_iter(
    params: ParameterDict,
    x_train: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
) -> Array:
    """predicts with GPR

    Predicts with GPR without instantiating the full matrix.
    The contraction with the linear coefficients is performed
    iteratively.
    """
    matvec = _Ax_lhs_fun(x1=x, x2=x_train, params=params, kernel=kernel, noise=False)
    mu = mu + matvec(c)
    return mu


@partial(jit, static_argnums=(7, 8))
def _predict_derivs_dense(
    params: ParameterDict,
    x_train: ArrayLike,
    jacobian_train: ArrayLike,
    x: ArrayLike,
    jacobian: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
    full_covariance: Optional[bool] = False,
) -> Union[Array, Tuple[Array, Array]]:
    """predicts derivative values with GPR

    Predicts the derivative values with GPR.
    This is a dense implementation: the full kernel is instantiated
    before contracting twith the linear coefficients.
    """
    K_mn = _A_derivs_lhs(
        x1=x_train,
        jacobian1=jacobian_train,
        x2=x,
        jacobian2=jacobian,
        params=params,
        kernel=kernel,
        noise=False,
    )
    mu = mu + jnp.dot(K_mn.T, c)

    if full_covariance:
        C_mm = _A_derivs_lhs(
            x1=x_train,
            jacobian1=jacobian_train,
            x2=x_train,
            jacobian2=jacobian_train,
            params=params,
            kernel=kernel,
            noise=True,
        )
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = _A_derivs_lhs(
            x1=x,
            jacobian1=jacobian,
            x2=x,
            jacobian2=jacobian,
            params=params,
            kernel=kernel,
            noise=False,
        )
        C_nn = C_nn - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn

    # recover the right shape
    ns, _, nd = jacobian.shape
    mu = mu.reshape(ns, nd)
    return mu


@partial(jit, static_argnums=(7,))
def _predict_derivs_iter(
    params: ParameterDict,
    x_train: ArrayLike,
    jacobian_train: ArrayLike,
    x: ArrayLike,
    jacobian: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
) -> Array:
    """predicts derivative values with GPR

    Predicts the derivative values with GPR.
    The contraction with the linear coefficients is performed
    iteratively.
    """
    matvec = _Ax_derivs_lhs_fun(
        x1=x,
        jacobian1=jacobian,
        x2=x_train,
        jacobian2=jacobian_train,
        params=params,
        kernel=kernel,
        noise=False,
    )
    mu = mu + matvec(c)
    # recover the right shape
    ns, _, nd = jacobian.shape
    mu = mu.reshape(ns, nd)
    return mu


# Functions to compute the log marginal likelihood


@partial(jit, static_argnums=(3, 4))
def _lml_dense(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Array:
    """log marginal likelihood for GPR

    Computes the log marginal likelihood for GPR.
    This is a dense implementation: the total kernel
    is built before obtaining the lml.
        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)
    """
    m = y.shape[0]
    mu = mean_function(y)
    y = y - mu

    C_mm = _A_lhs(x1=x, x2=x, params=params, kernel=kernel, noise=True)

    L_m = jsp.linalg.cholesky(C_mm, lower=True)
    cy = jsp.linalg.solve_triangular(L_m, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_m)))
    mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / m

    return mll


@partial(jit, static_argnums=(3, 4, 5, 6))
def _lml_iter(params, x, y, kernel, mean_function, num_evals, num_lanczos, lanczos_key):
    m = y.shape[0]
    c, mu = _fit_iter(
        params=params, x=x, y=y, kernel=kernel, mean_function=mean_function
    )
    y = y - mu

    matvec = _Ax_lhs_fun(x1=x, x2=x, params=params, kernel=kernel, noise=True)

    mll = -0.5 * jnp.sum(jnp.dot(y.T, c))
    mll -= 0.5 * lanczos_logdet(
        matvec,
        num_evals=int(num_evals),
        dim_mat=int(m),
        num_lanczos=int(num_lanczos),
        key=lanczos_key,
    )
    mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / m

    return mll


@partial(jit, static_argnums=(4, 5))
def _lml_derivs_dense(
    params: ParameterDict,
    x: ArrayLike,
    jacobian: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Array:
    """log marginal likelihood for GPR with derivative values

    Computes the log marginal likelihood for GPR when training
    on derivative values only.
    This is a dense implementation: the total kernel is built
    before obtaining the lml.

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)
    """
    kernel = partial(kernel.d01kj, jacobian1=jacobian, jacobian2=jacobian)
    # flatten y
    y = y.reshape(-1, 1)
    return _lml_dense(params, x, y, kernel, mean_function)


def _lml_derivs_iter(
    params, x, jacobian, y, kernel, mean_function, num_evals, num_lanczos, lanczos_key
):
    # flatten y
    y = y.reshape(-1, 1)
    m = y.shape[0]
    c, mu = _fit_derivs_iter(
        params=params,
        x=x,
        jacobian=jacobian,
        y=y,
        kernel=kernel,
        mean_function=mean_function,
    )
    y = y - mu

    matvec = _Ax_derivs_lhs_fun(
        x1=x,
        jacobian1=jacobian,
        x2=x,
        jacobian2=jacobian,
        params=params,
        kernel=kernel,
        noise=True,
    )

    mll = -0.5 * jnp.sum(jnp.dot(y.T, c))
    mll -= 0.5 * lanczos_logdet(
        matvec,
        num_evals=int(num_evals),
        dim_mat=int(m),
        num_lanczos=int(num_lanczos),
        key=lanczos_key,
    )
    mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / m

    return mll
