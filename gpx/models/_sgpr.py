from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit
from jax.scipy.sparse.linalg import cg
from jax.typing import ArrayLike

from ..kernels.kernels import Kernel
from ..lanczos import lanczos_logdet
from ..operations import rowfun_to_matvec
from ..parameters import Parameter

ParameterDict = Dict[str, Parameter]
KeyArray = Array

# Functions to compute the kernel matrices needed for SGPR


@partial(jit, static_argnums=(3,))
def _A_lhs(
    x1: ArrayLike, x2: ArrayLike, params: ParameterDict, kernel: Kernel
) -> Tuple[Array, Array]:
    """lhs and rhs of A∙x = b' = b∙y

    Builds the left hand side (lhs) and right hand side (rhs)
    of A∙x = b' = b∙y for SGPR.
    Dense implementation: A is built all at once.

        lhs = σ² K_mm + K_mn K_nm
        rhs = K_mn
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _ = x1.shape

    K_mn = kernel(x1, x2, kernel_params)
    C_mm = kernel(x1, x1, kernel_params)
    C_mm = sigma**2 * C_mm + jnp.dot(K_mn, K_mn.T) + 1e-10 * jnp.eye(m)

    return C_mm, K_mn


@partial(jit, static_argnums=(5,))
def _A_derivs_lhs(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
) -> Tuple[Array, Array]:
    """lhs and rhs of A∙x = b' = b∙y

    Builds the left hand side (lhs) and right hand side (rhs)
    of A∙x = b' = b∙y for SGPR.
    Dense implementation: A is built all at once.

        lhs = σ² K_mm + K_mn K_nm
        rhs = K_mn

    Where K = ∂₁∂₂K
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m = x1.shape[0] * jacobian1.shape[2]

    K_mn = kernel.d01kj(x1, x2, kernel_params, jacobian1, jacobian2)
    C_mm = kernel.d01kj(x1, x1, kernel_params, jacobian1, jacobian1)
    C_mm = sigma**2 * C_mm + jnp.dot(K_mn, K_mn.T) + 1e-10 * jnp.eye(m)

    return C_mm, K_mn


def _Ax_lhs_fun(
    x1: ArrayLike, x2: ArrayLike, params: ParameterDict, kernel: Kernel
) -> Tuple[Callable[ArrayLike, Array], Callable[ArrayLike, Array]]:
    """lhs and rhs of A∙x = b' = b∙y

    Builds the left hand side (lhs) and right hand side (rhs)
    of A∙x = b' = b∙y for SGPR.
    Iterative implementation.

        lhs = σ² K_mm + K_mn K_nm
        rhs = K_mn
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _ = x1.shape

    def row_fun_lhs(x1s):
        # σ² K_mm
        kernel_row_1 = kernel(x1s, x1, kernel_params) * sigma**2

        # K_mn∙K_nm
        def f(x2s):
            x2s = jnp.expand_dims(x2s, axis=0)
            a = kernel(x1s, x2, kernel_params)
            b = kernel(x2, x2s, kernel_params)
            return jnp.dot(a, b)

        kernel_row_2 = jax.vmap(f, in_axes=0, out_axes=2)(x1)
        kernel_row_2 = kernel_row_2.reshape(1, m)
        return kernel_row_1 + kernel_row_2

    def row_fun_rhs(x1s):
        return kernel(x1s, x2, kernel_params)

    jitter = 1e-10

    matvec_lhs = rowfun_to_matvec(
        row_fun_lhs, init_val=(x1,), update_diag=True, diag_value=jitter
    )
    matvec_rhs = rowfun_to_matvec(
        row_fun_rhs,
        init_val=(x1,),
        update_diag=False,
    )

    return matvec_lhs, matvec_rhs


def _Ax_derivs_lhs_fun(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
) -> Tuple[Callable[ArrayLike, Array], Callable[ArrayLike, Array]]:
    """lhs and rhs of A∙x = b' = b∙y

    Builds the left hand side (lhs) and right hand side (rhs)
    of A∙x = b' = b∙y for SGPR.
    Iterative implementation.

        lhs = σ² K_mm + K_mn K_nm
        rhs = K_mn

    where K = ∂₁∂₂K
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m = x1.shape[0]
    nd = jacobian1.shape[2]

    def row_fun_lhs(x1s, j1s):
        # σ² K_mm
        kernel_row_1 = kernel.d01kj(x1s, x1, kernel_params, j1s, jacobian1) * sigma**2

        # K_mn∙K_nm
        def f(x2s, j2s):
            x2s = jnp.expand_dims(x2s, axis=0)
            j2s = jnp.expand_dims(j2s, axis=0)
            a = kernel.d01kj(x1s, x2, kernel_params, j1s, jacobian2)
            b = kernel.d01kj(x2, x2s, kernel_params, jacobian2, j2s)
            res = jnp.dot(a, b)
            return res

        kernel_row_2 = jax.vmap(f, in_axes=(0, 0), out_axes=2)(x1, jacobian1)
        kernel_row_2 = kernel_row_2.reshape(nd, nd * m)
        return kernel_row_1 + kernel_row_2

    def row_fun_rhs(x1s, j1s):
        return kernel.d01kj(x1s, x2, kernel_params, j1s, jacobian2)

    jitter = 1e-10

    matvec_lhs = rowfun_to_matvec(
        row_fun_lhs, init_val=(x1, jacobian1), update_diag=True, diag_value=jitter
    )
    matvec_rhs = rowfun_to_matvec(
        row_fun_rhs,
        init_val=(x1, jacobian1),
        update_diag=False,
    )

    return matvec_lhs, matvec_rhs


def _Hx_fun(
    x1: ArrayLike, x2: ArrayLike, params: ParameterDict, kernel: Kernel
) -> Callable[ArrayLike, Array]:
    """builds H_nn = K_nm (K_mm)⁻¹ K_mn iteratively

    Builds the function computing H_nn∙x without ever instantiating
    H_nn. Only K_mm is instantiated fully and then inverted.
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _ = x1.shape
    n, _ = x2.shape
    Kmm = kernel(x1, x1, kernel_params) + 1e-10 * jnp.eye(m)
    Kmm_inv = jnp.linalg.inv(Kmm)

    def row_fun(x2s):
        kernel_row = kernel(x2s, x1, kernel_params)
        kernel_row = jnp.dot(kernel_row, Kmm_inv)

        def f(x3s):
            x3s = jnp.expand_dims(x3s, axis=0)
            res = jnp.dot(kernel_row, kernel(x1, x3s, kernel_params))
            return res

        kernel_row = jax.vmap(f, in_axes=0, out_axes=2)(x2)
        kernel_row = kernel_row.reshape(1, n)
        return kernel_row

    jitter_noise = sigma**2 + 1e-10

    matvec = rowfun_to_matvec(
        row_fun, init_val=(x2,), update_diag=True, diag_value=jitter_noise
    )

    return matvec


def _Hx_derivs_fun(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
) -> Callable[ArrayLike, Array]:
    """builds H_nn = K_nm (K_mm)⁻¹ K_mn iteratively

    Builds the function computing H_nn∙x without ever instantiating
    H_nn. Only K_mm is instantiated fully and then inverted.
    Here K = ∂₁∂₂K.
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _, nd = jacobian1.shape
    n = x2.shape[0]
    Kmm = kernel.d01kj(x1, x1, kernel_params, jacobian1, jacobian1) + 1e-10 * jnp.eye(
        m * nd
    )
    Kmm_inv = jnp.linalg.inv(Kmm)

    def row_fun(x2s, j2s):
        kernel_row = kernel.d01kj(x2s, x1, kernel_params, j2s, jacobian1)
        kernel_row = jnp.dot(kernel_row, Kmm_inv)

        def f(x3s, j3s):
            x3s = jnp.expand_dims(x3s, axis=0)
            j3s = jnp.expand_dims(j3s, axis=0)
            res = jnp.dot(
                kernel_row, kernel.d01kj(x1, x3s, kernel_params, jacobian1, j3s)
            )
            return res

        kernel_row = jax.vmap(f, in_axes=(0, 0), out_axes=2)(x2, jacobian2)
        kernel_row = kernel_row.reshape(nd, n * nd)
        return kernel_row

    jitter_noise = sigma**2 + 1e-10

    matvec = rowfun_to_matvec(
        row_fun, init_val=(x2, jacobian2), update_diag=True, diag_value=jitter_noise
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


# Functions to fit SGPR


@partial(jit, static_argnums=(4, 5))
def _fit_dense(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    Dense implementation.

    μ = m(y)
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    mu = mean_function(y)
    y = y - mu
    C_mm, K_mn = _A_lhs(x1=x_locs, x2=x, params=params, kernel=kernel)
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y))
    return c, mu


@partial(jit, static_argnums=(4, 5))
def _fit_iter(
    params: ParameterDict,
    x_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    Iterative implementation.

    μ = m(y)
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    mu = mean_function(y)
    y = y - mu
    matvec_lhs, matvec_rhs = _Ax_lhs_fun(x1=x_locs, x2=x, params=params, kernel=kernel)
    c, _ = cg(matvec_lhs, matvec_rhs(y))
    return c, mu


@partial(jit, static_argnums=(6, 7))
def _fit_derivs_dense(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    Dense implementation. Here K = ∂₁∂₂K.

    μ = 0.
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    y = y.reshape(-1, 1)
    mu = mean_function(y)
    y = y - mu
    C_mm, K_mn = _A_derivs_lhs(
        x1=x_locs,
        jacobian1=jacobian_locs,
        x2=x,
        jacobian2=jacobian,
        params=params,
        kernel=kernel,
    )
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y))
    return c, mu


@partial(jit, static_argnums=(6, 7))
def _fit_derivs_iter(
    params: ParameterDict,
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    Iterative implementation. Here K = ∂₁∂₂K.

    μ = 0.
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    y = y.reshape(-1, 1)
    mu = mean_function(y)
    y = y - mu
    matvec_lhs, matvec_rhs = _Ax_derivs_lhs_fun(
        x1=x_locs,
        jacobian1=jacobian_locs,
        x2=x,
        jacobian2=jacobian,
        params=params,
        kernel=kernel,
    )
    c, _ = cg(matvec_lhs, matvec_rhs(y))
    return c, mu


# Functions to predict with SGPR


@partial(jit, static_argnums=(5, 6))
def _predict_dense(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    m = x_locs.shape[0]

    K_nm = kernel(x, x_locs, kernel_params)
    mu = mu + jnp.dot(K_nm, c)

    if full_covariance:
        C_mm, K_mn = _A_lhs(x1=x_locs, x2=x, params=params, kernel=kernel)
        K_mm = kernel(x_locs, x_locs, kernel_params) + 1e-10 * jnp.eye(m)
        L_m = jsp.linalg.cholesky(K_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        H_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = kernel(x, x, kernel_params)
        C_nn = C_nn - jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.dot(H_mn.T, H_mn)
        return mu, C_nn

    return mu


@partial(jit, static_argnums=(7, 8))
def _predict_derivs_dense(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    jacobian: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
    jaccoef: Optional[ArrayLike] = None,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

    Here K = ∂₁∂₂K
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    if jaccoef is not None:
        # we have the contracted coefficients, so we try to be faster
        return mu + jnp.sum(
            kernel.d01kjc(x_locs, x, kernel_params, jaccoef, jacobian), axis=0
        )

    K_nm = kernel.d01kj(x, x_locs, kernel_params, jacobian, jacobian_locs)
    mu = mu + jnp.dot(K_nm, c)
    n, _, nd = jacobian.shape
    mu = mu.reshape(n, nd)

    if full_covariance:
        m = x_locs.shape[0]
        C_mm, K_mn = _A_derivs_lhs(
            x1=x_locs,
            jacobian1=jacobian_locs,
            x2=x,
            jacobian2=jacobian,
            params=params,
            kernel=kernel,
        )
        K_mm = kernel.d01kj(
            x_locs, x_locs, kernel_params, jacobian_locs, jacobian_locs
        ) + 1e-10 * jnp.eye(m)
        L_m = jsp.linalg.cholesky(K_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        H_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = kernel.d01kj(x, x, kernel_params, jacobian, jacobian)
        C_nn = C_nn - jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.dot(H_mn.T, H_mn)
        return mu, C_nn

    return mu


@partial(jit, static_argnums=(5,))
def _predict_iter(
    params: ParameterDict,
    x_locs: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn
    """
    _, matvec = _Ax_lhs_fun(x1=x, x2=x_locs, params=params, kernel=kernel)
    mu = mu + matvec(c)
    return mu


@partial(jit, static_argnums=(7,))
def _predict_derivs_iter(
    params: ParameterDict,
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    jacobian: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn

    Here K = ∂₁∂₂K
    """
    n, _, nd = jacobian.shape
    _, matvec = _Ax_derivs_lhs_fun(
        x1=x,
        jacobian1=jacobian,
        x2=x_locs,
        jacobian2=jacobian_locs,
        params=params,
        kernel=kernel,
    )
    mu = mu + matvec(c)
    mu = mu.reshape(n, nd)
    return mu


@partial(jit, static_argnums=[6, 7])
def _predict_y_derivs_dense(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
    jaccoef: Optional[ArrayLike] = None,
) -> Array:
    """predicts targets with a SGPR (projected processes)
    trained on derivatives.

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    m = x_locs.shape[0]

    if jaccoef is not None:
        # we have the contracted jacobian, so we try to be faster
        return mu + jnp.sum(kernel.d0kjc(x_locs, x, kernel_params, jaccoef), axis=0)

    K_mn = kernel.d0kj(x_locs, x, kernel_params, jacobian_locs)

    mu = mu + jnp.dot(K_mn.T, c)

    if full_covariance:
        C_mm = kernel.d01kj(x_locs, x_locs, kernel_params, jacobian_locs, jacobian_locs)

        C_mm = C_mm + sigma**2 * jnp.eye(m)
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)

        C_nn = kernel(x, x, kernel_params)

        C_nn = C_nn - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn

    return mu


@partial(jit, static_argnums=[6, 7])
def _predict_y_derivs_iter(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
    full_covariance: Optional[bool] = False,
) -> Array:
    """predicts targets with a SGPR (projected processes)
    trained on derivatives.

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn
    """
    matvec = _Kx_derivs1_fun(
        x1=x,
        x2=x_locs,
        jacobian2=jacobian_locs,
        params=params,
        kernel=kernel,
        noise=False,
    )
    mu = mu + matvec(c)
    return mu


# Functions to compute the log marginal likelihood for SGPR


@partial(jit, static_argnums=(4, 5))
def _lml_dense(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)
    H = K_nm (K_mm)⁻¹ K_mn
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m = x_locs.shape[0]
    mu = mean_function(y)
    y = y - mu
    n = y.shape[0]

    K_mm = kernel(x_locs, x_locs, kernel_params)
    K_mn = kernel(x_locs, x, kernel_params)

    L_m = jsp.linalg.cholesky(K_mm + 1e-10 * jnp.eye(m), lower=True)
    G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
    C_nn = jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.eye(n) + 1e-10 * jnp.eye(n)
    L_n = jsp.linalg.cholesky(C_nn, lower=True)
    cy = jsp.linalg.solve_triangular(L_n, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_n)))
    mll -= n * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / n

    return mll


@partial(jit, static_argnums=(4, 5, 6, 7))
def _lml_iter(
    params: ParameterDict,
    x_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    num_evals: int,
    num_lanczos: int,
    lanczos_key: KeyArray,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    Iterative implementation.

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)
    H = K_nm (K_mm)⁻¹ K_mn
    """
    mu = mean_function(y)
    y = y - mu
    n = y.shape[0]

    matvec = _Hx_fun(x_locs, x, params, kernel)

    mll = -0.5 * jnp.sum(jnp.dot(y.T, cg(matvec, y)[0]))
    mll -= 0.5 * lanczos_logdet(
        matvec,
        num_evals=int(num_evals),
        dim_mat=int(n),
        num_lanczos=int(num_lanczos),
        key=lanczos_key,
    )
    mll -= n * 0.5 * jnp.log(2.0 * jnp.pi)

    mll = mll / n

    return mll


@partial(jit, static_argnums=(6, 7))
def _lml_derivs_dense(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    Dense implementation.

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)
    H = K_nm (K_mm)⁻¹ K_mn

    Here K = ∂₁∂₂K
    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m = x_locs.shape[0]
    mu = mean_function(y)
    y = y - mu
    n = y.shape[0]

    K_mm = kernel.d01kj(x_locs, x_locs, kernel_params, jacobian_locs, jacobian_locs)
    K_mn = kernel.d01kj(x_locs, x, kernel_params, jacobian_locs, jacobian)

    L_m = jsp.linalg.cholesky(K_mm + 1e-10 * jnp.eye(m), lower=True)
    G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
    C_nn = jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.eye(n) + 1e-10 * jnp.eye(n)
    L_n = jsp.linalg.cholesky(C_nn, lower=True)
    cy = jsp.linalg.solve_triangular(L_n, y, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_n)))
    mll -= n * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / n

    return mll


@partial(jit, static_argnums=(6, 7, 8, 9))
def _lml_derivs_iter(
    params: ParameterDict,
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    num_evals: int,
    num_lanczos: int,
    lanczos_key: KeyArray,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    Iterative implementation.

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)
    H = K_nm (K_mm)⁻¹ K_mn

    Here K = ∂₁∂₂K
    """
    y = y.reshape(-1, 1)
    mu = mean_function(y)
    y = y - mu
    n = y.shape[0]

    matvec = _Hx_derivs_fun(x_locs, jacobian_locs, x, jacobian, params, kernel)

    mll = -0.5 * jnp.sum(jnp.dot(y.T, cg(matvec, y)[0]))
    mll -= 0.5 * lanczos_logdet(
        matvec,
        num_evals=int(num_evals),
        dim_mat=int(n),
        num_lanczos=int(num_lanczos),
        key=lanczos_key,
    )
    mll -= n * 0.5 * jnp.log(2.0 * jnp.pi)

    mll = mll / n

    return mll
