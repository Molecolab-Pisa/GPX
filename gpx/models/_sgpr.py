from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit
from jax._src import prng
from jax.scipy.sparse.linalg import cg
from jax.typing import ArrayLike

from ..kernels.kernels import Kernel
from ..lanczos import lanczos_logdet
from ..parameters import Parameter

ParameterDict = Dict[str, Parameter]

# Functions to compute the kernel matrices needed for SGPR


def _A_lhs(
    x1: ArrayLike, x2: ArrayLike, params: ParameterDict, kernel: Kernel
) -> Tuple[Array, Array]:
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _ = x1.shape

    K_mn = kernel(x1, x2, kernel_params)
    C_mm = kernel(x1, x1, kernel_params)
    C_mm = sigma**2 * C_mm + jnp.dot(K_mn, K_mn.T) + 1e-10 * jnp.eye(m)

    return C_mm, K_mn


def _A_derivs_lhs(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
) -> Tuple[Array, Array]:
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
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    @jit
    def matvec_lhs(z):
        def update_row(carry, x1s):
            # σ² K_mm
            kernel_row_1 = (
                kernel(x1s[jnp.newaxis], x1, kernel_params).squeeze(axis=0) * sigma**2
            )
            # K_mn K_nm
            kernel_row_2 = jax.vmap(
                lambda x2s: jnp.dot(
                    kernel(x1s[jnp.newaxis], x2, kernel_params),
                    kernel(x2, x2s[jnp.newaxis], kernel_params),
                ).squeeze()
            )(x1)
            kernel_row = kernel_row_1 + kernel_row_2
            kernel_row = kernel_row.at[carry].add(1e-10)
            rowvec = jnp.dot(kernel_row, z)
            carry = carry + 1
            return carry, rowvec

        _, res = jax.lax.scan(update_row, 0, x1)
        return res

    @jit
    def matvec_rhs(z):
        def update_row(carry, x1s):
            kernel_row = kernel(x1s[jnp.newaxis], x2, kernel_params)
            rowvec = jnp.dot(kernel_row, z)
            return carry, rowvec

        _, res = jax.lax.scan(update_row, 0, x1)
        return res

    return matvec_lhs, matvec_rhs


def _Ax_derivs_lhs_fun(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
) -> Tuple[Callable[ArrayLike, Array], Callable[ArrayLike, Array]]:
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m = x1.shape[0]
    nd = jacobian1.shape[2]

    @jit
    def matvec_lhs(z):
        def update_row(carry, xj):
            x1s, j1s = xj
            # σ² K_mm
            kernel_row_1 = (
                kernel.d01kj(
                    x1s[jnp.newaxis], x1, kernel_params, j1s[jnp.newaxis], jacobian1
                )
                * sigma**2
            )

            # K_mn K_nm
            def f(x2s, j2s):
                a = kernel.d01kj(
                    x1s[jnp.newaxis], x2, kernel_params, j1s[jnp.newaxis], jacobian2
                )
                b = kernel.d01kj(
                    x2, x2s[jnp.newaxis], kernel_params, jacobian2, j2s[jnp.newaxis]
                )
                res = jnp.dot(a, b)
                return res

            kernel_row_2 = jax.vmap(f, in_axes=(0, 0), out_axes=2)(x1, jacobian1)
            kernel_row_2 = kernel_row_2.reshape(nd, nd * m)
            kernel_row = kernel_row_1 + kernel_row_2
            # we have to add the noise + jitter to the diagonal
            # we do so by updating the stripe block by block, where
            # the block has the dimension of the number of rows in the
            # slice
            start_indices = (0, nd * carry)
            jnoise = (1e-10) * jnp.eye(nd)
            fill = jax.lax.dynamic_slice(kernel_row, start_indices, (nd, nd)) + jnoise
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

    @jit
    def matvec_rhs(z):
        def update_row(carry, xj):
            x1s, j1s = xj
            kernel_row = kernel.d01kj(
                x1s[jnp.newaxis], x2, kernel_params, j1s[jnp.newaxis], jacobian2
            )
            rowvec = jnp.dot(kernel_row, z)
            return carry, rowvec

        _, res = jax.lax.scan(update_row, 0, (x1, jacobian1))
        res = jnp.concatenate(res, axis=0)
        return res

    return matvec_lhs, matvec_rhs


def _Hx_fun(
    x1: ArrayLike, x2: ArrayLike, params: ParameterDict, kernel: Kernel
) -> Callable[ArrayLike, Array]:
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _ = x1.shape
    Kmm = kernel(x1, x1, kernel_params) + 1e-10 * jnp.eye(m)
    Kmm_inv = jnp.linalg.inv(Kmm)

    @jit
    def matvec(z):
        def update_row(carry, x2s):
            kernel_row = kernel(x2s[jnp.newaxis], x1, kernel_params)
            kernel_row = jnp.dot(kernel_row, Kmm_inv)

            def f(x3s):
                res = jnp.dot(kernel_row, kernel(x1, x3s[jnp.newaxis], kernel_params))
                return res.squeeze()

            kernel_row = jax.vmap(f)(x2)
            kernel_row = kernel_row.at[carry].add(sigma**2 + 1e-10)
            rowvec = jnp.dot(kernel_row, z)
            carry = carry + 1
            return carry, rowvec

        _, res = jax.lax.scan(update_row, 0, x2)
        return res

    return matvec


def _Hx_derivs_fun(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
) -> Callable[ArrayLike, Array]:
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value

    m, _, nd = jacobian1.shape
    n = x2.shape[0]
    Kmm = kernel.d01kj(x1, x1, kernel_params, jacobian1, jacobian1) + 1e-10 * jnp.eye(
        m * nd
    )
    Kmm_inv = jnp.linalg.inv(Kmm)

    @jit
    def matvec(z):
        def update_row(carry, x2j):
            x2s, j2s = x2j
            kernel_row = kernel.d01kj(
                x2s[jnp.newaxis], x1, kernel_params, j2s[jnp.newaxis], jacobian1
            )
            kernel_row = jnp.dot(kernel_row, Kmm_inv)

            def f(x3s, j3s):
                res = jnp.dot(
                    kernel_row,
                    kernel.d01kj(
                        x1, x3s[jnp.newaxis], kernel_params, jacobian1, j3s[jnp.newaxis]
                    ),
                )
                return res

            kernel_row = jax.vmap(f, in_axes=(0, 0), out_axes=2)(x2, jacobian2)
            kernel_row = kernel_row.reshape(nd, n * nd)
            # we have to add the noise + jitter to the diagonal
            # we do so by updating the stripe block by block, where
            # the block has the dimension of the number of rows in the
            # slice
            start_indices = (0, nd * carry)
            jnoise = (sigma**2 + 1e-10) * jnp.eye(nd)
            fill = jax.lax.dynamic_slice(kernel_row, start_indices, (nd, nd)) + jnoise
            kernel_row = jax.lax.dynamic_update_slice(
                kernel_row,
                fill,
                start_indices,
            )
            rowvec = jnp.dot(kernel_row, z)
            carry = carry + 1
            return carry, rowvec

        _, res = jax.lax.scan(update_row, 0, (x2, jacobian2))
        res = jnp.concatenate(res, axis=0)
        return res

    return matvec


# Functions to fit SGPR


@partial(jit, static_argnums=(3, 4))
def _fit_dense(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    μ = m(y)
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    x_locs = params["x_locs"].value
    mu = mean_function(y)
    y = y - mu
    C_mm, K_mn = _A_lhs(x1=x_locs, x2=x, params=params, kernel=kernel)
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y))
    return c, mu


@partial(jit, static_argnums=(3, 4))
def _fit_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    x_locs = params["x_locs"].value
    mu = mean_function(y)
    y = y - mu
    matvec_lhs, matvec_rhs = _Ax_lhs_fun(x1=x_locs, x2=x, params=params, kernel=kernel)
    c, _ = cg(matvec_lhs, matvec_rhs(y))
    return c, mu


@partial(jit, static_argnums=(4, 5))
def _fit_derivs_dense(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    jacobian: ArrayLike,
    mean_function: Callable,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    μ = 0.
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value
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


@partial(jit, static_argnums=(4, 5))
def _fit_derivs_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    jacobian: ArrayLike,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value
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


@partial(jit, static_argnums=(4, 5))
def _predict_dense(
    params: Dict[str, Parameter],
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
    x_locs = params["x_locs"].value
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


@partial(jit, static_argnums=(5, 6))
def _predict_derivs_dense(
    params: Dict[str, Parameter],
    x: ArrayLike,
    jacobian: ArrayLike,
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
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value

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


@partial(jit, static_argnums=(4,))
def _predict_iter(
    params: ParameterDict, x: ArrayLike, c: ArrayLike, mu: ArrayLike, kernel: Kernel
) -> Array:
    x_locs = params["x_locs"].value
    _, matvec = _Ax_lhs_fun(x1=x, x2=x_locs, params=params, kernel=kernel)
    mu = mu + matvec(c)
    return mu


@partial(jit, static_argnums=(5,))
def _predict_derivs_iter(
    params: ParameterDict,
    x: ArrayLike,
    jacobian: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
) -> Array:
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value
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


# Functions to compute the log marginal likelihood for SGPR


@partial(jit, static_argnums=(3, 4))
def _lml_dense(
    params: Dict[str, Parameter],
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
    x_locs = params["x_locs"].value

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


@partial(jit, static_argnums=(3, 4, 5, 6))
def _lml_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    num_evals: int,
    num_lanczos: int,
    lanczos_key: prng.PRNGKeyArray,
) -> Array:
    x_locs = params["x_locs"].value

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


@partial(jit, static_argnums=(4, 5))
def _lml_derivs_dense(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

    H = K_nm (K_mm)⁻¹ K_mn

    """
    kernel_params = params["kernel_params"]
    sigma = params["sigma"].value
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value

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


@partial(jit, static_argnums=(4, 5, 6, 7))
def _lml_derivs_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    num_evals: int,
    num_lanczos: int,
    lanczos_key: prng.PRNGKeyArray,
) -> Array:
    x_locs = params["x_locs"].value
    jacobian_locs = params["jacobian_locs"].value

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
