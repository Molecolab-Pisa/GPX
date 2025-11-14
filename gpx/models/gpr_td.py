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

import warnings
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array, jit, lax, random
from jax.scipy.sparse.linalg import cg
from jax.typing import ArrayLike

from ..defaults import gpxargs
from ..lanczos import lanczos_logdet
from ..mean_functions import zero_mean
from ..operations import recover_first_axis, update_row_diagonal
from ..optimizers.scipy_optimize import scipy_minimize_ol
from ..parameters import ModelState, Parameter
from .utils import loss_fn_with_args, randomized_minimization_ol

ParameterDict = Dict[str, Parameter]
Kernel = Any
KeyArray = Array


@partial(jit, static_argnums=(7, 8, 9))
def _A_lhs(
    x1: ArrayLike,
    jacobian1_1: ArrayLike,
    jacobian1_2: ArrayLike,
    x2: ArrayLike,
    jacobian2_1: ArrayLike,
    jacobian2_2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
    predict: Optional[bool] = False,
) -> Array:
    """lhs of A x = b

    Builds the left hand side (lhs) of A x = b for GPR.
    Dense implementation: A is built all at once.

            / K + σ²I     ∂₂K       \
        A = |                       |
            \   ∂₁K    ∂₁∂₂K + σ²I  /

    This includes also a second derivative if given.
    """
    kernel_params = params["kernel_params"]
    sigma_targets = params["sigma_targets"].value
    sigma_derivs = params["sigma_derivs"].value
    if jacobian1_2 is not None:
        sigma_derivs2 = params["sigma_derivs2"].value

    # build kernel with target and derivatives

    K = kernel.k(x1=x1, x2=x2, params=kernel_params, predict=predict)
    if noise:
        K = K + (sigma_targets**2 + 1e-10) * jnp.eye(K.shape[0])

    D01kj_1 = kernel.d01kj(
        x1=x1,
        jacobian1=jacobian1_1,
        x2=x2,
        jacobian2=jacobian2_1,
        params=kernel_params,
        predict=predict,
    )
    if noise:
        D01kj_1 = D01kj_1 + (sigma_derivs**2 + 1e-10) * jnp.eye(D01kj_1.shape[0])

    D0kj_1 = kernel.d0kj(
        x1=x1,
        x2=x2,
        params=kernel_params,
        jacobian=jacobian1_1,
        predict=predict,
    )

    D1kj_1 = kernel.d1kj(
        x1=x1,
        x2=x2,
        params=kernel_params,
        jacobian=jacobian2_1,
        predict=predict,
    )

    if jacobian1_2 is None:
        C_mm = jnp.concatenate(
            (
                jnp.concatenate((K, D1kj_1), axis=1),
                jnp.concatenate((D0kj_1, D01kj_1), axis=1),
            ),
            axis=0,
        )
    else:

        D01kj_2 = kernel.d01kj(
            x1=x1,
            jacobian1=jacobian1_2,
            x2=x2,
            jacobian2=jacobian2_2,
            params=kernel_params,
            predict=predict,
        )
        if noise:
            D01kj_2 = D01kj_2 + (sigma_derivs2**2 + 1e-10) * jnp.eye(D01kj_2.shape[0])

        D01kj_12 = kernel.d01kj(
            x1=x1,
            jacobian1=jacobian1_1,
            x2=x2,
            jacobian2=jacobian2_2,
            params=kernel_params,
            predict=predict,
        )

        D01kj_21 = kernel.d01kj(
            x1=x1,
            jacobian1=jacobian1_2,
            x2=x2,
            jacobian2=jacobian2_1,
            params=kernel_params,
            predict=predict,
        )

        D0kj_2 = kernel.d0kj(
            x1=x1,
            x2=x2,
            params=kernel_params,
            jacobian=jacobian1_2,
            predict=predict,
        )

        D1kj_2 = kernel.d1kj(
            x1=x1,
            x2=x2,
            params=kernel_params,
            jacobian=jacobian2_2,
            predict=predict,
        )

        C_mm = jnp.concatenate(
            (
                jnp.concatenate((K, D1kj_1, D1kj_2), axis=1),
                jnp.concatenate((D0kj_1, D01kj_1, D01kj_12), axis=1),
                jnp.concatenate((D0kj_2, D01kj_21, D01kj_2), axis=1),
            ),
            axis=0,
        )
    return C_mm


def _Ax_lhs_fun(
    x1: ArrayLike,
    jacobian1_1: ArrayLike,
    jacobian1_2: ArrayLike,
    x2: ArrayLike,
    jacobian2_1: ArrayLike,
    jacobian2_2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
    predict: Optional[bool] = False,
) -> Callable[ArrayLike, Array]:
    """matrix-vector function for the lhs of A x = b

    Builds a function that computes the matrix-vector
    product of the left hand side (lhs) of A x = b.
    Iterative implementation.

            / K + σ²I     ∂₂K       \
        A = |                       |
            \   ∂₁K    ∂₁∂₂K + σ²I  /

    This includes also a second derivative if given.
    """
    kernel_params = params["kernel_params"]
    sigma_targets = params["sigma_targets"].value
    sigma_derivs = params["sigma_derivs"].value

    if kernel.nperms is not None:
        nperms = kernel.nperms
        nsp2, _, _ = jacobian2_1.shape
        ns2 = int(nsp2 / nperms)
        if not predict:
            nsp1, nf1, jv1_1 = jacobian1_1.shape
            ns1 = int(nsp1 / nperms)
            x1 = x1.reshape(ns1, nperms, nf1)
            jacobian1_1 = jacobian1_1.reshape(ns1, nperms, nf1, jv1_1)
            _, _, jv2_1 = jacobian2_1.shape
            if jacobian1_2 is not None:
                _, _, jv1_2 = jacobian1_2.shape
                jacobian1_2 = jacobian1_2.reshape(ns1, nperms, nf1, jv1_2)
                _, _, jv2_2 = jacobian2_2.shape
        else:
            ns1, _ = x1.shape
            _, _, jv1_1 = jacobian1_1.shape
            _, _, jv2_1 = jacobian2_1.shape
            if jacobian1_2 is not None:
                _, _, jv1_2 = jacobian1_2.shape
                _, _, jv2_2 = jacobian2_2.shape

    else:
        ns1, _, jv1_1 = jacobian1_1.shape
        ns2, _, _ = jacobian2_1.shape
        if jacobian1_2 is not None:
            _, _, jv1_2 = jacobian1_2.shape
            _, _, jv2_2 = jacobian2_2.shape

    @jit
    def matvec(z):

        _, _, jv2_1 = jacobian2_1.shape

        z1 = z[:ns2]
        z2 = z[ns2 : ns2 + ns2 * jv2_1]
        if jacobian2_2 is not None:
            _, _, jv2_2 = jacobian2_2.shape
            sigma_derivs2 = params["sigma_derivs2"].value
            z3 = z[-ns2 * jv2_2 :]

        def update_row(carry, xjs):

            if jacobian1_2 is not None:
                x1s, j1s, j2s = recover_first_axis(xjs)
                if kernel.nperms is not None:
                    if not predict:
                        x1s = x1s[0]
                        j1s = j1s[0]
                        j2s = j2s[0]
            else:
                x1s, j1s = recover_first_axis(xjs)
                if kernel.nperms is not None:
                    if not predict:
                        x1s = x1s[0]
                        j1s = j1s[0]

            #            if kernel.nperms is not None:
            row11 = kernel.k(x2, x1s, kernel_params, predict=predict).T
            row12 = kernel.d0kj(x2, x1s, kernel_params, jacobian2_1, predict=predict).T
            if noise:
                jitter_noise1 = sigma_targets**2 + 1e-10
                row11 = update_row_diagonal(row11, carry, jitter_noise1)
            if jacobian2_2 is not None:
                row13 = kernel.d0kj(
                    x2, x1s, kernel_params, jacobian2_2, predict=predict
                ).T
                rowvec1 = jnp.dot(row11, z1) + jnp.dot(row12, z2) + jnp.dot(row13, z3)
            else:
                rowvec1 = jnp.dot(row11, z1) + jnp.dot(row12, z2)

            row21 = kernel.d1kj(x2, x1s, kernel_params, j1s, predict=predict).T
            row22 = kernel.d01kj(
                x2, x1s, kernel_params, jacobian2_1, j1s, predict=predict
            ).T
            if noise:
                jitter_noise2 = sigma_derivs**2 + 1e-10
                row22 = update_row_diagonal(row22, carry, jitter_noise2)
            if jacobian2_2 is not None:
                row23 = kernel.d01kj(
                    x2, x1s, kernel_params, jacobian2_2, j1s, predict=predict
                ).T
                rowvec2 = jnp.dot(row21, z1) + jnp.dot(row22, z2) + jnp.dot(row23, z3)
            else:
                rowvec2 = jnp.dot(row21, z1) + jnp.dot(row22, z2)

            if jacobian1_2 is not None:
                row31 = kernel.d1kj(x2, x1s, kernel_params, j2s, predict=predict).T
                row32 = kernel.d01kj(
                    x2, x1s, kernel_params, jacobian2_1, j2s, predict=predict
                ).T
                row33 = kernel.d01kj(
                    x2, x1s, kernel_params, jacobian2_2, j2s, predict=predict
                ).T
                if noise:
                    jitter_noise2 = sigma_derivs2**2 + 1e-10
                    row33 = update_row_diagonal(row33, carry, jitter_noise2)
                rowvec3 = jnp.dot(row31, z1) + jnp.dot(row32, z2) + jnp.dot(row33, z3)
                rowvec = jnp.concatenate((rowvec1, rowvec2, rowvec3), axis=0)
            else:
                rowvec = jnp.concatenate((rowvec1, rowvec2), axis=0)

            carry = carry + 1

            return carry, rowvec

        if jacobian1_2 is not None:
            _, res = lax.scan(update_row, 0, (x1, jacobian1_1, jacobian1_2))
        else:
            _, res = lax.scan(update_row, 0, (x1, jacobian1_1))
        res = jnp.concatenate(res, axis=0)

        if jacobian1_2 is not None:
            mask = (
                np.array(
                    [
                        [
                            1,
                        ]
                        + [0 for x in range(jv1_1)]
                        + [0 for x in range(jv1_2)]
                        for x in range(ns1)
                    ]
                )
                .reshape(-1)
                .astype(bool)
            )
            ind = np.arange(ns1 + ns1 * jv1_1 + ns1 * jv1_2)[mask]
            res1 = res[ind]
            mask = (
                np.array(
                    [
                        [
                            0,
                        ]
                        + [1 for x in range(jv1_1)]
                        + [0 for x in range(jv1_2)]
                        for x in range(ns1)
                    ]
                )
                .reshape(-1)
                .astype(bool)
            )
            ind = np.arange(ns1 + ns1 * jv1_1 + ns1 * jv1_2)[mask]
            res2 = res[ind]
            mask = (
                np.array(
                    [
                        [
                            0,
                        ]
                        + [0 for x in range(jv1_1)]
                        + [1 for x in range(jv1_2)]
                        for x in range(ns1)
                    ]
                )
                .reshape(-1)
                .astype(bool)
            )
            ind = np.arange(ns1 + ns1 * jv1_1 + ns1 * jv1_2)[mask]
            res3 = res[ind]
            return jnp.concatenate((res1, res2, res3))
        else:
            res1 = res[jnp.arange(0, ns1 + ns1 * jv1_1, jv1_1 + 1)]
            res2 = jnp.delete(res, np.arange(0, ns1 + ns1 * jv1_1, jv1_1 + 1), axis=0)
            return jnp.concatenate((res1, res2))

    return matvec


@partial(jit, static_argnums=(4, 5))
def rpcholesky_TD(
    key: KeyArray,
    x: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    n_pivots: int,
    kernel: Kernel,
    kernel_params: Dict[str, Parameter],
) -> Tuple[Array, Array]:
    """Randomly Pivoted Cholesky decomposition
            K_nystr = K(:, S) K(S, S)^-1 K(S, :)

            / k + σ²I     ∂₂k       \
        K = |                       |
            \   ∂₁k    ∂₁∂₂k + σ²I  /

    K includes also a second derivative if given.
    """
    # x is (number of points, number of features)
    if kernel.nperms is not None:
        nperms = kernel.nperms
        np, nf, jv_1 = jacobian.shape
        n = int(np / nperms)
        x = x.reshape(n, nperms, nf)
        jacobian = jacobian.reshape(n, nperms, nf, jv_1)
    else:
        n, _ = x.shape
        _, _, jv_1 = jacobian.shape

    if jacobian_2 is not None:
        _, _, jv_2 = jacobian_2.shape
        if kernel.nperms is not None:
            jacobian_2 = jacobian_2.reshape(n, nperms, nf, jv_2)
        # factor matrix (F)
        fmat = jnp.zeros((n + n * jv_1 + n * jv_2, n_pivots))
    else:
        # factor matrix (F)
        fmat = jnp.zeros((n + n * jv_1, n_pivots))

    # list of pivot indices (S)
    pivots = jnp.zeros((n_pivots,), dtype=int)

    # compute the diagonal
    def diagonal_wrapper(init, xjs):
        # ensure that each point is (1,)
        if jacobian_2 is not None:
            xs, j1s, j2s = xjs
            if kernel.nperms is None:
                xs = jnp.expand_dims(xs, axis=0)
                j1s = jnp.expand_dims(j1s, axis=0)
                j2s = jnp.expand_dims(j2s, axis=0)
            return init, (
                kernel.k(xs, xs, kernel_params).squeeze(),
                kernel.d01kj(xs, xs, kernel_params, j1s, j1s),
                kernel.d01kj(xs, xs, kernel_params, j2s, j2s),
            )
        else:
            xs, j1s = xjs
            if kernel.nperms is None:
                xs = jnp.expand_dims(xs, axis=0)
                j1s = jnp.expand_dims(j1s, axis=0)
            return init, (
                kernel(xs, xs, kernel_params).squeeze(),
                kernel.d01kj(xs, xs, kernel_params, j1s, j1s),
            )

    if jacobian_2 is not None:
        _, diags = lax.scan(diagonal_wrapper, init=0, xs=(x, jacobian, jacobian_2))
        diag = jnp.concatenate(
            (
                diags[0],
                jnp.diagonal(diags[1], axis1=1, axis2=2).reshape(-1),
                jnp.diagonal(diags[2], axis1=1, axis2=2).reshape(-1),
            )
        )
    else:
        _, diags = lax.scan(diagonal_wrapper, init=0, xs=(x, jacobian))
        diag = jnp.concatenate(
            (diags[0], jnp.diagonal(diags[1], axis1=1, axis2=2).reshape(-1))
        )

    # iteratively build the Nystrom approximation
    def fori_wrapper(i, val):
        fmat, diag, pivots, key = val

        # sample s ~ d / sum(d)
        prob = diag / jnp.sum(diag)
        key, subkey = random.split(key)
        s = random.choice(key, jnp.arange(prob.shape[0]), p=prob, shape=())

        # update the pivots
        pivots = pivots.at[i].set(s)

        # compute the schur complement g
        # note: we are unable to select only a few columns of F due to
        #       the compilation within lax.fori_loop

        def first_row():
            if kernel.nperms is None:
                xs = jnp.expand_dims(x[s], axis=0)
                k = kernel.k(xs, x, kernel_params).squeeze()
                d1kj1 = kernel.d1kj(xs, x, kernel_params, jacobian).squeeze()
                if jacobian_2 is not None:
                    d1kj2 = kernel.d1kj(xs, x, kernel_params, jacobian_2).squeeze()
                    return jnp.concatenate((k, d1kj1, d1kj2), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )
                else:
                    return jnp.concatenate((k, d1kj1), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )
            else:
                xs = x[s]
                k = kernel.k(xs, x.reshape(np, nf), kernel_params).squeeze()
                d1kj1 = kernel.d1kj(
                    xs, x.reshape(np, nf), kernel_params, jacobian.reshape(np, nf, jv_1)
                ).squeeze()
                if jacobian_2 is not None:
                    d1kj2 = kernel.d1kj(
                        xs,
                        x.reshape(np, nf),
                        kernel_params,
                        jacobian_2.reshape(np, nf, jv_2),
                    ).squeeze()
                    return jnp.concatenate((k, d1kj1, d1kj2), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )
                else:
                    return jnp.concatenate((k, d1kj1), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )

        def second_row():
            index1 = ((s - n) / jv_1).astype(int)
            index2 = jnp.mod((s - n), jv_1)
            if kernel.nperms is None:
                xs = jnp.expand_dims(x[index1], axis=0)
                j1s = jnp.expand_dims(jacobian[index1], axis=0)
                d0kj = kernel.d0kj(xs, x, kernel_params, j1s)[index2]
                d01kj1 = kernel.d01kj(xs, x, kernel_params, j1s, jacobian)[index2]
                if jacobian_2 is not None:
                    d01kj2 = kernel.d01kj(xs, x, kernel_params, j1s, jacobian_2)[index2]
                    return jnp.concatenate((d0kj, d01kj1, d01kj2), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )
                else:
                    return jnp.concatenate((d0kj, d01kj1), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )
            else:
                xs = x[index1]
                j1s = jacobian[index1]
                d0kj = kernel.d0kj(xs, x.reshape(np, nf), kernel_params, j1s)[index2]
                d01kj1 = kernel.d01kj(
                    xs,
                    x.reshape(np, nf),
                    kernel_params,
                    j1s,
                    jacobian.reshape(np, nf, jv_1),
                )[index2]
                if jacobian_2 is not None:
                    d01kj2 = kernel.d01kj(
                        xs,
                        x.reshape(np, nf),
                        kernel_params,
                        j1s,
                        jacobian_2.reshape(np, nf, jv_2),
                    )[index2]
                    return jnp.concatenate((d0kj, d01kj1, d01kj2), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )
                else:
                    return jnp.concatenate((d0kj, d01kj1), axis=0) - jnp.dot(
                        fmat, fmat[s].T
                    )

        def third_row():
            index1 = ((s - n - n * jv_1) / jv_2).astype(int)
            index2 = jnp.mod((s - n - n * jv_1), jv_2)
            if kernel.nperms is None:
                xs = jnp.expand_dims(x[index1], axis=0)
                j2s = jnp.expand_dims(jacobian_2[index1], axis=0)
                d0kj = kernel.d0kj(xs, x, kernel_params, j2s)[index2]
                d01kj1 = kernel.d01kj(xs, x, kernel_params, j2s, jacobian)[index2]
                d01kj2 = kernel.d01kj(xs, x, kernel_params, j2s, jacobian_2)[index2]
                return jnp.concatenate((d0kj, d01kj1, d01kj2), axis=0) - jnp.dot(
                    fmat, fmat[s].T
                )
            else:
                xs = x[index1]
                j2s = jacobian_2[index1]
                d0kj = kernel.d0kj(xs, x.reshape(np, nf), kernel_params, j2s)[index2]
                d01kj1 = kernel.d01kj(
                    xs,
                    x.reshape(np, nf),
                    kernel_params,
                    j2s,
                    jacobian.reshape(np, nf, jv_1),
                )[index2]
                d01kj2 = kernel.d01kj(
                    xs,
                    x.reshape(np, nf),
                    kernel_params,
                    j2s,
                    jacobian_2.reshape(np, nf, jv_2),
                )[index2]
                return jnp.concatenate((d0kj, d01kj1, d01kj2), axis=0) - jnp.dot(
                    fmat, fmat[s].T
                )

        def false_fun():
            if jacobian_2 is not None:
                return jnp.where(s < n + n * jv_1, second_row(), third_row())
            else:
                return second_row()

        g = jnp.where(s < n, first_row(), false_fun())

        # update the i-th column of the factor matrix
        # note: 1e-6 is a jitter factor that ensures we are not dividing by 0
        fmat = fmat.at[:, i].set(g / (g[s] ** 0.5 + 1e-6))

        # update the diagonal
        diag = diag - fmat[:, i] ** 2

        return (fmat, diag, pivots, key)

    res = lax.fori_loop(0, n_pivots, fori_wrapper, init_val=(fmat, diag, pivots, key))

    fmat, _, pivots, _ = res

    return fmat, pivots


def _precond_rpcholesky_TD(
    x: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    n_pivots: int,
    key: KeyArray,
) -> Callable[ArrayLike, Array]:

    kernel_params = params["kernel_params"]
    sigma_targets = params["sigma_targets"].value

    fmat, _ = rpcholesky_TD(
        key=key,
        x=x,
        jacobian=jacobian,
        jacobian_2=jacobian_2,
        n_pivots=n_pivots,
        kernel=kernel,
        kernel_params=kernel_params,
    )

    P_kk = sigma_targets**2 * jnp.eye(n_pivots) + fmat.T @ fmat
    P_kk = jnp.linalg.inv(P_kk)

    @jit
    def matvec(z):
        res = jnp.dot(fmat.T, z)
        res = jnp.dot(P_kk, res)
        res = -(sigma_targets ** (-2)) * jnp.dot(fmat, res)

        res = res + sigma_targets ** (-2) * z
        return res

    return matvec


@partial(jit, static_argnums=(7, 8))
def _lml_dense(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    y_derivs: ArrayLike,
    y_derivs_2: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Array:
    kernel_params = params["kernel_params"]
    sigma_targets = params["sigma_targets"].value
    sigma_derivs = params["sigma_derivs"].value

    mu = mean_function(y)
    y = y - mu
    y = y.reshape(-1, 1)
    y_derivs = y_derivs.reshape(-1, 1)
    y_m = jnp.concatenate((y, y_derivs))
    m = y_m.shape[0]
    if y_derivs_2 is not None:
        sigma_derivs2 = params["sigma_derivs2"].value
        y_derivs_2 = y_derivs_2.reshape(-1, 1)
        y_m = jnp.concatenate((y_m, y_derivs_2))

    # build kernel with target and derivatives
    K = kernel(x1=x, x2=x, params=kernel_params)
    K = K + sigma_targets**2 * jnp.eye(K.shape[0])

    D01kj_1 = kernel.d01kj(
        x1=x,
        jacobian1=jacobian,
        x2=x,
        jacobian2=jacobian,
        params=kernel_params,
    )
    D01kj_1 = D01kj_1 + sigma_derivs**2 * jnp.eye(D01kj_1.shape[0])

    D0kj_1 = kernel.d0kj(
        x1=x,
        x2=x,
        params=kernel_params,
        jacobian=jacobian,
    )

    if jacobian_2 is None:
        C_mm = jnp.concatenate(
            (
                jnp.concatenate((K, D0kj_1.T), axis=1),
                jnp.concatenate((D0kj_1, D01kj_1), axis=1),
            ),
            axis=0,
        )
    else:
        D01kj_2 = kernel.d01kj(
            x1=x,
            jacobian1=jacobian_2,
            x2=x,
            jacobian2=jacobian_2,
            params=kernel_params,
        )
        D01kj_2 = D01kj_2 + sigma_derivs2**2 * jnp.eye(D01kj_2.shape[0])

        D01kj_12 = kernel.d01kj(
            x1=x,
            jacobian1=jacobian,
            x2=x,
            jacobian2=jacobian_2,
            params=kernel_params,
        )

        D0kj_2 = kernel.d0kj(
            x1=x,
            x2=x,
            params=kernel_params,
            jacobian=jacobian_2,
        )

        C_mm = jnp.concatenate(
            (
                jnp.concatenate((K, D0kj_1.T, D0kj_2.T), axis=1),
                jnp.concatenate((D0kj_1, D01kj_1, D01kj_12), axis=1),
                jnp.concatenate((D0kj_2, D01kj_12.T, D01kj_2), axis=1),
            ),
            axis=0,
        )
    L_m = jsp.linalg.cholesky(C_mm, lower=True)
    cy = jsp.linalg.solve_triangular(L_m, y_m, lower=True)

    mll = -0.5 * jnp.sum(jnp.square(cy))
    mll -= jnp.sum(jnp.log(jnp.diag(L_m)))
    mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / m

    return mll


@partial(jit, static_argnums=(7, 8, 9, 10, 12))
def _lml_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    y_derivs: ArrayLike,
    y_derivs_2: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    num_evals: int,
    num_lanczos: int,
    key_lanczos: KeyArray,
    n_pivots: int,
    key_precond: KeyArray,
):
    """log marginal likelihood for GPR

    Computes the log marginal likelihood for GPR when training
    on targets and derivatives.
    Iterative implementation: the kernel is never instantiated, and
    the log|K| is estimated via stochastic trace estimation and lanczos
    quadrature.

        lml = - ½ y^T (K_nn + σ²I)⁻¹ y - ½ log |K_nn + σ²I| - ½ n log(2π)
    """
    c, mu = _fit_iter(
        params=params,
        x=x,
        jacobian=jacobian,
        jacobian_2=jacobian_2,
        y=y,
        y_derivs=y_derivs,
        y_derivs_2=y_derivs_2,
        kernel=kernel,
        mean_function=mean_function,
        n_pivots=n_pivots,
        key_precond=key_precond,
    )
    y = y - mu
    y = y.reshape(-1, 1)
    y_derivs = y_derivs.reshape(-1, 1)
    y_m = jnp.concatenate((y, y_derivs))
    if y_derivs_2 is not None:
        y_derivs_2 = y_derivs_2.reshape(-1, 1)
        y_m = jnp.concatenate((y_m, y_derivs_2))
    m = y_m.shape[0]

    matvec = _Ax_lhs_fun(
        x1=x,
        jacobian1_1=jacobian,
        jacobian1_2=jacobian_2,
        x2=x,
        jacobian2_1=jacobian,
        jacobian2_2=jacobian_2,
        params=params,
        kernel=kernel,
        noise=True,
    )

    mll = -0.5 * jnp.sum(jnp.dot(y_m.T, c))
    mll -= 0.5 * lanczos_logdet(
        matvec,
        num_evals=int(num_evals),
        dim_mat=int(m),
        num_lanczos=int(num_lanczos),
        key=key_lanczos,
    )
    mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)

    # normalize by the number of samples
    mll = mll / m

    return mll


@partial(jit, static_argnums=(7, 8))
def _fit_dense(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    y_derivs: ArrayLike,
    y_derivs_2: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
) -> Tuple[Array, Array]:
    """fits a GPR on targets and derivatives
    with Cholesky

    Fits a GPR on targets and derivatives.
    The linear system is solved using the
    Cholesky decomposition.

    μ = m(y)
    c = (K + σ²I)⁻¹(y - μ)
    """
    mu = mean_function(y)
    y = y - mu
    y = y.reshape(-1, 1)
    y_derivs = y_derivs.reshape(-1, 1)
    y_m = jnp.concatenate((y, y_derivs))
    if y_derivs_2 is not None:
        y_derivs_2 = y_derivs_2.reshape(-1, 1)
        y_m = jnp.concatenate((y_m, y_derivs_2))

    C_mm = _A_lhs(
        x1=x,
        jacobian1_1=jacobian,
        jacobian1_2=jacobian_2,
        x2=x,
        jacobian2_1=jacobian,
        jacobian2_2=jacobian_2,
        params=params,
        kernel=kernel,
        noise=True,
    )
    c = jnp.linalg.solve(C_mm, y_m)
    return c, mu


@partial(jit, static_argnums=(7, 8, 9))
def _fit_iter(
    params: ParameterDict,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    y_derivs: ArrayLike,
    y_derivs_2: ArrayLike,
    kernel: Kernel,
    mean_function: Callable[ArrayLike, Array],
    n_pivots: int,
    key_precond: KeyArray,
    c_guess: ArrayLike = None,
) -> Tuple[Array, Array]:
    """fits a GPR on targets and
    derivatives iteratively

    Fits a GPR on targets and derivatives solving
    the linear system iteratively with
    Conjugated Gradient.

    μ = m(y)
    c = (K + σ²I)⁻¹(y - μ)
    """
    mu = mean_function(y)
    y = y - mu
    y = y.reshape(-1, 1)
    y_derivs = y_derivs.reshape(-1, 1)
    y_m = jnp.concatenate((y, y_derivs))
    if y_derivs_2 is not None:
        y_derivs_2 = y_derivs_2.reshape(-1, 1)
        y_m = jnp.concatenate((y_m, y_derivs_2))

    matvec = _Ax_lhs_fun(
        x1=x,
        jacobian1_1=jacobian,
        jacobian1_2=jacobian_2,
        x2=x,
        jacobian2_1=jacobian,
        jacobian2_2=jacobian_2,
        params=params,
        kernel=kernel,
        noise=True,
    )

    precond = _precond_rpcholesky_TD(
        x=x,
        jacobian=jacobian,
        jacobian_2=jacobian_2,
        params=params,
        kernel=kernel,
        n_pivots=n_pivots,
        key=key_precond,
    )

    if c_guess is not None:
        c, _ = cg(matvec, y_m, M=precond, atol=1e-7, x0=c_guess)
    else:
        c, _ = cg(matvec, y_m, M=precond, atol=1e-7)

    return c, mu


@partial(jit, static_argnums=(9))
def _predict_dense(
    params: ParameterDict,
    x_train: ArrayLike,
    jacobian_train: ArrayLike,
    jacobian_train_2: ArrayLike,
    x: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
) -> Union[Array, Tuple[Array, Array]]:
    """predicts target and derivative values with GPR

    Predicts with GPR, by first building the full kernel matrix
    and then contracting with the linear coefficients.

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)
    """
    K_mn = _A_lhs(
        x1=x_train,
        jacobian1_1=jacobian_train,
        jacobian1_2=jacobian_train_2,
        x2=x,
        jacobian2_1=jacobian,
        jacobian2_2=jacobian_2,
        params=params,
        kernel=kernel,
        noise=False,
        predict=True,
    )

    pred = jnp.dot(K_mn.T, c)

    ns, _, nv_1 = jacobian.shape

    pred = pred.at[:ns, :].add(mu)
    y_pred = pred[:ns]
    y_derivs_pred = pred[ns : ns + nv_1 * ns].reshape(ns, -1)

    if jacobian_2 is not None:
        _, _, nv_2 = jacobian_2.shape
        y_derivs_pred_2 = pred[ns + nv_1 * ns : ns + nv_1 * ns + nv_2 * ns].reshape(
            ns, -1
        )
        return y_pred, y_derivs_pred, y_derivs_pred_2
    else:
        return y_pred, y_derivs_pred


@partial(jit, static_argnums=(9))
def _predict_iter(
    params: ParameterDict,
    x_train: ArrayLike,
    jacobian_train: ArrayLike,
    jacobian_train_2: ArrayLike,
    x: ArrayLike,
    jacobian: ArrayLike,
    jacobian_2: ArrayLike,
    c: ArrayLike,
    mu: ArrayLike,
    kernel: Kernel,
):
    """predicts target and derivative values with GPR

    Predicts target and derivative values with GPR.
    The contraction with the linear coefficients is performed
    iteratively.

    μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)
    """
    matvec = _Ax_lhs_fun(
        x1=x,
        jacobian1_1=jacobian,
        jacobian1_2=jacobian_2,
        x2=x_train,
        jacobian2_1=jacobian_train,
        jacobian2_2=jacobian_train_2,
        params=params,
        kernel=kernel,
        noise=False,
        predict=True,
    )
    pred = matvec(c)
    # recover the right shape
    ns, _, nv_1 = jacobian.shape

    pred = pred.at[:ns, :].add(mu)
    y_pred = pred[:ns]
    y_derivs_pred = pred[ns : ns + nv_1 * ns].reshape(ns, -1)

    if jacobian_2 is not None:
        _, _, nv_2 = jacobian_2.shape
        y_derivs_pred_2 = pred[ns + nv_1 * ns : ns + nv_1 * ns + nv_2 * ns].reshape(
            ns, -1
        )
        return y_pred, y_derivs_pred, y_derivs_pred_2
    else:
        return y_pred, y_derivs_pred


def log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    y_derivs: ArrayLike,
    jacobian_2: ArrayLike = None,
    y_derivs_2: ArrayLike = None,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    key_lanczos: Optional[KeyArray] = gpxargs.key_lanczos,
    n_pivots: Optional[int] = None,
    key_precond: Optional[KeyArray] = None,
) -> Array:
    """computes the log marginal likelihood for standard gaussian process
    using the kernel, the Hessian kernel and the off diagonal blocks.

        lml = - ½ y^T (∂₁∂₂K_nn + σ²I)⁻¹ y - ½ log |∂₁∂₂K_nn + σ²I| - ½ n log(2π)

    Args:
        state: model state
        x: observations
        y: labels
        jacobian/jacobian_2: jacobian of x
        y_derivs/y_derivs_2: label derivatives
        iterative: whether to compute the lml iteratively
                   (e.g., never instantiating the kernel)
        num_evals: number of monte carlo evaluations for estimating
                   log|K| (used only if iterative=True)
        num_lanczos: number of Lanczos evaluations for estimating
                     log|K| (used only if iterative=True)
        key_lanczos: random key for Lanczos tridiagonalization
    Returns:
        lml: log marginal likelihood
    """
    if iterative:
        return _lml_iter(
            params=state.params,
            x=x,
            y=y,
            jacobian=jacobian,
            jacobian_2=jacobian_2,
            y_derivs=y_derivs,
            y_derivs_2=y_derivs_2,
            kernel=state.kernel,
            mean_function=state.mean_function,
            num_evals=num_evals,
            num_lanczos=num_lanczos,
            key_lanczos=key_lanczos,
            n_pivots=n_pivots,
            key_precond=key_precond,
        )
    return _lml_dense(
        params=state.params,
        x=x,
        y=y,
        jacobian=jacobian,
        jacobian_2=jacobian_2,
        y_derivs=y_derivs,
        y_derivs_2=y_derivs_2,
        kernel=state.kernel,
        mean_function=state.mean_function,
    )


def neg_log_marginal_likelihood(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    y_derivs: ArrayLike,
    jacobian_2: ArrayLike = None,
    y_derivs_2: ArrayLike = None,
    iterative: Optional[bool] = False,
    num_evals: Optional[int] = gpxargs.num_evals,
    num_lanczos: Optional[int] = gpxargs.num_lanczos,
    key_lanczos: Optional[KeyArray] = gpxargs.key_lanczos,
    n_pivots: Optional[int] = None,
    key_precond: Optional[KeyArray] = None,
) -> Array:
    "Returns the negative log marginal likelihood"
    return -log_marginal_likelihood(
        state=state,
        x=x,
        y=y,
        jacobian=jacobian,
        jacobian_2=jacobian_2,
        y_derivs=y_derivs,
        y_derivs_2=y_derivs_2,
        iterative=iterative,
        num_evals=num_evals,
        num_lanczos=num_lanczos,
        key_lanczos=key_lanczos,
        n_pivots=n_pivots,
        key_precond=key_precond,
    )


class GPR_TD:
    def __init__(
        self,
        kernel: Kernel,
        mean_function: Callable = zero_mean,
        kernel_params: Dict[str, Parameter] = None,
        sigma_targets: Parameter = None,
        sigma_derivs: Parameter = None,
        sigma_derivs2: Parameter = None,
        loss_fn: Callable = neg_log_marginal_likelihood,
    ) -> None:

        params = {
            "kernel_params": kernel_params,
            "sigma_targets": sigma_targets,
            "sigma_derivs": sigma_derivs,
            "sigma_derivs2": sigma_derivs2,
        }
        opt = {
            "x_train": None,
            "jacobian_train": None,
            "jacobian_train_2": None,
            "jaccoef": None,
            "jaccoef_2": None,
            "y_train": None,
            "y_derivs_train": None,
            "y_derivs_train_2": None,
            "is_fitted": False,
            "c": None,
            "c_targets": None,
            "mu": None,
        }

        self.state = ModelState(kernel, mean_function, params, loss_fn=loss_fn, **opt)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        jacobian: ArrayLike,
        y_derivs: ArrayLike,
        jacobian_2: ArrayLike = None,
        y_derivs_2: ArrayLike = None,
        iterative: Optional[bool] = False,
        key: Optional[KeyArray] = None,
        num_restarts: Optional[int] = 0,
        minimize: Optional[bool] = True,
        return_history: Optional[bool] = False,
        loss_kwargs: Optional[Dict] = None,
        n_pivots: Optional[int] = None,
        key_precond: Optional[KeyArray] = None,
        c_guess: Optional[Array] = None,
    ) -> ModelState:
        """fits a gaussian process on targets and
        derivatives

            μ = m(y)
            c = (K_nn + σ²I)⁻¹y

        Args:
            state: model state
            x: observations
            y: labels
            jacobian/jacobian_2: jacobian of x
            y_derivs/y_derivs_2
            iterative: whether to fit iteratively
                       (e.g., never instantiating the kernel)
        Returns:
            state: fitted model state
        """
        if loss_kwargs is None:
            loss_kwargs = {}
        loss_kwargs["iterative"] = iterative
        loss_kwargs["n_pivots"] = n_pivots
        loss_kwargs["key_precond"] = gpxargs.key_precond
        loss_fn = loss_fn_with_args(self.state.loss_fn, loss_kwargs)

        if minimize:
            minimization_function = scipy_minimize_ol
            self.state, optres, *history = randomized_minimization_ol(
                key=key,
                state=self.state,
                x=x,
                y=y,
                jacobian=jacobian,
                y_derivs=y_derivs,
                jacobian_2=jacobian_2,
                y_derivs_2=y_derivs_2,
                loss_fn=loss_fn,
                minimization_function=minimization_function,
                num_restarts=num_restarts,
                return_history=return_history,
            )
            self.optimize_results_ = optres

            # if the optimization is failed, print a warning
            if not optres.success:
                warnings.warn(
                    "optimization returned with error: {:d}. ({:s})".format(
                        optres.status, optres.message
                    ),
                    stacklevel=2,
                )

        fit_func = (
            partial(
                _fit_iter,
                n_pivots=n_pivots,
                key_precond=gpxargs.key_precond,
                c_guess=c_guess,
            )
            if iterative
            else _fit_dense
        )

        c, mu = fit_func(
            params=self.state.params,
            x=x,
            y=y,
            jacobian=jacobian,
            jacobian_2=jacobian_2,
            y_derivs=y_derivs,
            y_derivs_2=y_derivs_2,
            kernel=self.state.kernel,
            mean_function=self.state.mean_function,
        )
        if self.state.kernel.nperms is not None:
            nperms = self.state.kernel.nperms
            nsp, nf, nv_1 = jacobian.shape
            ns = int(nsp / nperms)
        else:
            ns, _, nv_1 = jacobian.shape

        c_targets = c[:ns].reshape(-1)
        c_derivs1 = c[ns : ns + nv_1 * ns]
        if self.state.kernel.nperms is not None:
            jaccoef = jnp.einsum(
                "sv,spfv->spf",
                c_derivs1.reshape(ns, nv_1),
                jacobian.reshape(ns, nperms, nf, nv_1),
            ).reshape(ns * nperms, nf)
        else:
            jaccoef = jnp.einsum("sv,sfv->sf", c_derivs1.reshape(ns, nv_1), jacobian)
        if jacobian_2 is not None:
            _, _, nv_2 = jacobian_2.shape
            c_derivs2 = c[ns + nv_1 * ns : ns + nv_1 * ns + nv_2 * ns]
            if self.state.kernel.nperms is not None:
                jaccoef_2 = jnp.einsum(
                    "sv,spfv->spf",
                    c_derivs2.reshape(ns, nv_2),
                    jacobian_2.reshape(ns, nperms, nf, nv_2),
                ).reshape(ns * nperms, nf)
            else:
                jaccoef_2 = jnp.einsum(
                    "sv,sfv->sf", c_derivs2.reshape(ns, nv_2), jacobian_2
                )
        else:
            jaccoef_2 = None

        self.state = self.state.update(
            dict(
                x_train=x,
                y_train=y,
                jacobian_train=jacobian,
                jacobian_train_2=jacobian_2,
                jaccoef=jaccoef,
                jaccoef_2=jaccoef_2,
                y_derivs_train=y_derivs,
                y_derivs_train_2=y_derivs_2,
                c=c,
                c_targets=c_targets,
                mu=mu,
                is_fitted=True,
            )
        )

        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self

    def predict(
        self,
        x: ArrayLike,
        jacobian: ArrayLike,
        jacobian_2: ArrayLike = None,
        iterative: Optional[bool] = False,
    ):
        """predicts targets and derivatives with gaussian process

            μ = K_nm (K_mm + σ²)⁻¹y

        Args:
            state: model state
            x: observations
            jacobian/jacobian_2: jacobian of x
            full_covariance: whether to return the covariance matrix too
            iterative: whether to fit iteratively
                       (e.g., never instantiating the kernel)
        Returns:
            μ: predicted mean
        """
        if not self.state.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Run `fit` to fit the model before prediction."
            )
        if iterative:
            return _predict_iter(
                params=self.state.params,
                x_train=self.state.x_train,
                jacobian_train=self.state.jacobian_train,
                jacobian_train_2=self.state.jacobian_train_2,
                x=x,
                jacobian=jacobian,
                jacobian_2=jacobian_2,
                c=self.state.c,
                mu=self.state.mu,
                kernel=self.state.kernel,
            )

        return _predict_dense(
            params=self.state.params,
            x_train=self.state.x_train,
            jacobian_train=self.state.jacobian_train,
            jacobian_train_2=self.state.jacobian_train_2,
            x=x,
            jacobian=jacobian,
            jacobian_2=jacobian_2,
            c=self.state.c,
            mu=self.state.mu,
            kernel=self.state.kernel,
        )

    def print(self) -> None:
        return self.state.print_params()

    def save(self, state_file):
        return self.state.save(state_file)

    def load(self, state_file):
        self.state = self.state.load(state_file)
        return self
