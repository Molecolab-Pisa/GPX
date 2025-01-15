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

from functools import partial
from typing import Callable, Dict, Optional, Tuple

from jax import Array, jit
from jax.typing import ArrayLike

from ..kernels.approximations import rpcholesky
from ..kernels.kernels import Kernel
from ..parameters import Parameter
from . import _sgpr

ParameterDict = Dict[str, Parameter]
KeyArray = Array


# functions to fit a SGPR


@partial(jit, static_argnums=(4, 5, 6))
def _fit_dense(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    key: KeyArray,
    n_locs: int,
    kernel: Callable,
    mean_function: Callable,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    μ = m(y)
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    c, mu = _sgpr._fit_dense(
        params=params,
        x_locs=x_locs,
        x=x,
        y=y,
        kernel=kernel,
        mean_function=mean_function,
    )
    return c, mu, x_locs


@partial(jit, static_argnums=(4, 5, 6))
def _fit_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    key: KeyArray,
    n_locs: int,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    c, mu = _sgpr._fit_iter(
        params=params,
        x_locs=x_locs,
        x=x,
        y=y,
        kernel=kernel,
        mean_function=mean_function,
    )
    return c, mu, x_locs


@partial(jit, static_argnums=(5, 6, 7))
def _fit_derivs_dense(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    key: KeyArray,
    n_locs: int,
    kernel: Callable,
    mean_function: Callable,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    μ = 0.
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    jacobian_locs = jacobian[pivots].copy()
    c, mu = _sgpr._fit_derivs_dense(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        y=y,
        jacobian=jacobian,
        kernel=kernel,
        mean_function=mean_function,
    )
    return c, mu, x_locs, jacobian_locs


@partial(jit, static_argnums=(5, 6, 7))
def _fit_derivs_iter(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    key: KeyArray,
    n_locs: int,
    kernel: Callable,
    mean_function: Callable,
) -> Tuple[Array, Array]:
    """fits a SGPR (projected processes)

    μ = 0.
    c = (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    """
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    jacobian_locs = jacobian[pivots].copy()
    c, mu = _sgpr._fit_derivs_iter(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        y=y,
        jacobian=jacobian,
        kernel=kernel,
        mean_function=mean_function,
    )
    return c, mu, x_locs, jacobian_locs


# functions to compute the log marginal likelihood in SGPR


@partial(jit, static_argnums=(4, 5, 6))
def _lml_dense(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    key: KeyArray,
    n_locs: int,
    kernel: Callable,
    mean_function: Callable,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)
    H = K_nm (K_mm)⁻¹ K_mn
    """
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    return _sgpr._lml_dense(
        params=params,
        x_locs=x_locs,
        x=x,
        y=y,
        kernel=kernel,
        mean_function=mean_function,
    )


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def _lml_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    key: KeyArray,
    n_locs: int,
    kernel: Callable,
    mean_function: Callable,
    num_evals: int,
    num_lanczos: int,
    key_lanczos: KeyArray,
):
    # note: this is a dense operation, and as is
    #       is not sufficient to iteratively evaluate
    #       the lml
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    return _sgpr._lml_iter(
        params=params,
        x_locs=x_locs,
        x=x,
        y=y,
        kernel=kernel,
        mean_function=mean_function,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        key_lanczos=key_lanczos,
    )


@partial(jit, static_argnums=(5, 6, 7))
def _lml_derivs_dense(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    key: KeyArray,
    n_locs: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

    H = K_nm (K_mm)⁻¹ K_mn

    """
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    jacobian_locs = jacobian[pivots].copy()
    return _sgpr._lml_derivs_dense(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        y=y,
        jacobian=jacobian,
        kernel=kernel,
        mean_function=mean_function,
    )


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
    return _sgpr._predict_dense(
        params=params,
        x_locs=x_locs,
        x=x,
        c=c,
        mu=mu,
        kernel=kernel,
        full_covariance=full_covariance,
    )


@partial(jit, static_argnums=(5,))
def _predict_iter(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    x: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
) -> Array:
    """predicts with a SGPR (projected processes)

    μ = K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn y
    C_nn = K_nn - K_nm (K_mm)⁻¹ K_mn + σ² K_nm (σ² K_mm + K_mn K_nm)⁻¹ K_mn
    """
    return _sgpr._predict_iter(
        params=params,
        x_locs=x_locs,
        x=x,
        c=c,
        mu=mu,
        kernel=kernel,
    )


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
    return _sgpr._predict_derivs_dense(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        jacobian=jacobian,
        c=c,
        mu=mu,
        kernel=kernel,
        full_covariance=full_covariance,
        jaccoef=jaccoef,
    )


@partial(jit, static_argnums=(7,))
def _predict_derivs_iter(
    params: Dict[str, Parameter],
    x_locs: ArrayLike,
    jacobian_locs: ArrayLike,
    x: ArrayLike,
    jacobian: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Callable,
) -> Array:
    return _sgpr._predict_derivs_iter(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        jacobian=jacobian,
        c=c,
        mu=mu,
        kernel=kernel,
    )


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
    return _sgpr._predict_y_derivs_dense(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        c=c,
        mu=mu,
        kernel=kernel,
        jaccoef=jaccoef,
    )


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
    return _sgpr._predict_y_derivs_iter(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        c=c,
        mu=mu,
        kernel=kernel,
    )


@partial(jit, static_argnums=(5, 6, 7, 8, 9))
def _lml_derivs_iter(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    key: KeyArray,
    n_locs: ArrayLike,
    kernel: Callable,
    mean_function: Callable,
    num_evals: int,
    num_lanczos: int,
    key_lanczos: KeyArray,
) -> Array:
    """log marginal likelihood for SGPR (projected processes)

    lml = - ½ y^T (H + σ²)⁻¹ y - ½ log|H + σ²| - ½ n log(2π)

    H = K_nm (K_mm)⁻¹ K_mn

    """
    _, pivots = rpcholesky(
        key=key,
        x=x,
        n_pivots=n_locs,
        kernel=kernel,
        kernel_params=params["kernel_params"],
    )
    x_locs = x[pivots].copy()
    jacobian_locs = jacobian[pivots].copy()
    return _sgpr._lml_derivs_iter(
        params=params,
        x_locs=x_locs,
        jacobian_locs=jacobian_locs,
        x=x,
        y=y,
        jacobian=jacobian,
        kernel=kernel,
        mean_function=mean_function,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        key_lanczos=key_lanczos,
    )
