from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit
from jax.scipy.sparse.linalg import cg
from jax.typing import ArrayLike

from ..lanczos import lanczos_logdet
from ..operations import rowfun_to_matvec
from ..parameters import Parameter

ParameterDict = Dict[str, Parameter]
Kernel = Any
KeyArray = Array

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

        A = K + σ²I
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
    """lhs of A x = b

    Builds the left hand side (lhs) of A x = b for GPR.
    Dense implementation: A is built all at once.

        A = ∂₁∂₂K + σ²I
    """
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
    Iterative implementation.

        A = K + σ²I
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    def row_fun(x1s):
        return kernel(x1s, x2, kernel_params)

    jitter_noise = sigma**2 + 1e-10

    return rowfun_to_matvec(
        row_fun, init_val=(x1,), update_diag=noise, diag_value=jitter_noise
    )


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
    Iterative implementation.

        A = ∂₁∂₂K + σ²I
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    def row_fun(x1s, j1s):
        return kernel.d01kj(x1s, x2, kernel_params, j1s, jacobian2)

    jitter_noise = sigma**2 + 1e-10

    matvec = rowfun_to_matvec(
        row_fun, init_val=(x1, jacobian1), update_diag=noise, diag_value=jitter_noise
    )

    return matvec


def _Kx_derivs1_fun(
    x1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
) -> Callable[ArrayLike, Array]:
    """matrix-vector function for the first derivative of K

    Builds a function that computes the matrix-vector
    product ∂₂K x.
    Iterative implementation.
    """
    kernel_params = params["kernel_params"]

    def row_fun(x1s):
        x1s = jnp.expand_dims(x1s, axis=0)
        return kernel.d1kj(x1s, x2, kernel_params, jacobian2)

    matvec = rowfun_to_matvec(row_fun, init_val=(x1))

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

    Fits a standard GPR. The linear system is solved using the
    Cholesky decomposition.

    μ = m(y)
    c = (K + σ²I)⁻¹(y - μ)
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
    c = (K + σ²I)⁻¹(y - μ)
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
    c = (∂₁∂₂K + σ²I)⁻¹(y - μ)
    """
    # also flatten y
    y = y.reshape(-1, 1)
    mu = mean_function(y)
    y = y - mu
    C_mm = _A_derivs_lhs(
        x1=x,
        jacobian1=jacobian,
        x2=x,
        jacobian2=jacobian,
        params=params,
        kernel=kernel,
        noise=True,
    )
    c = jnp.linalg.solve(C_mm, y)
    return c, mu


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
    c = (∂₁∂₂K + σ²I)⁻¹(y - μ)
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

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)
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

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)
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
    jaccoef: Optional[ArrayLike] = None,
) -> Union[Array, Tuple[Array, Array]]:
    """predicts derivative values with GPR

    Predicts the derivative values with GPR.
    This is a dense implementation: the full kernel is instantiated
    before contracting twith the linear coefficients.

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)
    C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn

    where K = ∂₁∂₂K
    """
    ns, _, nd = jacobian.shape

    # we have the contracted jacobian, so we try to be faster
    # note that this is incompatible with full_covariance as we
    # do not have the kernel
    if jaccoef is not None:
        mu = mu + jnp.sum(
            kernel.d01kjc(
                x1=x_train,
                x2=x,
                params=params["kernel_params"],
                jaccoef=jaccoef,
                jacobian=jacobian,
            ),
            axis=0,
        )
        return mu.reshape(ns, nd)

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

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)

    where K = ∂₁∂₂K
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


@partial(jit, static_argnums=[6, 7])
def _predict_y_derivs_dense(
    params: Dict[str, Parameter],
    x_train: ArrayLike,
    jacobian_train: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
    jaccoef: Optional[ArrayLike] = None,
) -> Array:
    """predicts targets with GPR trained on derivatives

    Predicts with GPR, by first building the full kernel matrix
    and then contracting with the linear coefficients.

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)
    C_nn = K_nn - K_nm (K_mm + σ²I)⁻¹ K_mn
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    if jaccoef is not None:
        # we have the contracted jacobian so we try to be faster
        n = x.shape[0]
        o = c.shape[1]
        mu = mu + jnp.sum(
            kernel.d0kjc(x1=x_train, x2=x, params=kernel_params, jaccoef=jaccoef),
            axis=0,
        )
        return mu.reshape(n, o)

    K_mn = kernel.d0kj(x_train, x, kernel_params, jacobian_train)

    mu = mu + jnp.dot(K_mn.T, c)

    if full_covariance:
        C_mm = kernel.d01kj(
            x_train, x_train, kernel_params, jacobian_train, jacobian_train
        )

        C_mm = C_mm + sigma**2 * jnp.eye(K_mn.shape[0])
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)

        C_nn = kernel(x, x, kernel_params)

        C_nn = C_nn - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn

    return mu


@partial(jit, static_argnums=[6, 7])
def _predict_y_derivs_iter(
    params: Dict[str, Parameter],
    x_train: ArrayLike,
    jacobian_train: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
) -> Array:
    """predicts targets with GPR trained on derivatives

    Predicts with GPR without instantiating the full matrix.
    The contraction with the linear coefficients is performed
    iteratively.

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)
    """
    matvec = _Kx_derivs1_fun(
        x1=x,
        x2=x_train,
        jacobian2=jacobian_train,
        params=params,
        kernel=kernel,
        noise=False,
    )
    mu = mu + matvec(c)
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
def _lml_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    num_evals: int,
    num_lanczos: int,
    lanczos_key: KeyArray,
):
    """log marginal likelihood for GPR

    Computes the log marginal likelihood for GPR.
    Iterative implementation: the kernel is never instantiated, and
    the log|K| is estimated via stochastic trace estimation and lanczos
    quadrature.

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)
    """
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

    where K = ∂₁∂₂K
    """
    # flatten y
    y = y.reshape(-1, 1)
    m = y.shape[0]
    mu = mean_function(y)
    y = y - mu

    C_mm = _A_derivs_lhs(
        x1=x,
        x2=x,
        jacobian1=jacobian,
        jacobian2=jacobian,
        params=params,
        kernel=kernel,
        noise=True,
    )

    L_m = jsp.linalg.cholesky(C_mm, lower=True)
    cy = jsp.linalg.solve_triangular(L_m, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_m)))
    mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / m

    return mll


@partial(jit, static_argnums=(4, 5, 6, 7))
def _lml_derivs_iter(
    params: ParameterDict,
    x: ArrayLike,
    jacobian: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    num_evals: int,
    num_lanczos: int,
    lanczos_key: KeyArray,
) -> Array:
    """log marginal likelihood for GPR

    Computes the log marginal likelihood for GPR.
    Iterative implementation: the kernel is never instantiated, and
    the log|K| is estimated via stochastic trace estimation and lanczos
    quadrature.

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)

    where K = ∂₁∂₂K
    """
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


@partial(jit, static_argnums=[5, 6])
def _mse(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    y_derivs: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
    coeff: float,
) -> Array:
    """Computes the mean squared loss for the target and its derivative

    mse = 1/N*Σ_i(μ_i - y_i)**2 + coeff * 1/N * 1/Natoms * Σ_j||∂μ_j/∂x - ∂y_j/∂x||**2

    """
    kernel_params = params["kernel_params"]

    mu = mean_function(y)
    y = y - mu
    _, _, jv = y_derivs.shape
    y_derivs = y_derivs.reshape((-1, 1))
    y_tilde = jnp.concatenate((y, y_derivs))

    C_mm = _A_lhs(x1=x, x2=x, params=params, kernel=kernel, noise=True)
    C_derivs = kernel.d0kj(x, x, kernel_params, jacobian)

    C_tilde = jnp.concatenate((C_mm, C_derivs))

    U, s, Vt = jnp.linalg.svd(C_tilde, full_matrices=False)

    Aeff = jnp.diag(s) @ Vt
    beff = U.T @ y_tilde

    c = jnp.linalg.solve(Aeff, beff).reshape(-1, 1)

    K_mm = _A_lhs(x1=x, x2=x, params=params, kernel=kernel, noise=False)
    mu = jnp.dot(c.T, K_mm).reshape(-1, 1)

    K_nm = kernel.d1kj(x, x, kernel_params, jacobian)
    y_pred_derivs = jnp.dot(c.T, K_nm)

    loss = ((mu - y) ** 2).mean() + coeff * (
        jnp.sum(
            (y_pred_derivs.reshape(-1, jv) - y_derivs.reshape(-1, jv)) ** 2, axis=-1
        ).mean()
    )

    return loss
