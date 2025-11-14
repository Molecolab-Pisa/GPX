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
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.typing import ArrayLike

Parameter = Any
KeyArray = Array


@partial(jit, static_argnums=(2, 3))
def rpcholesky(
    key: KeyArray,
    x: ArrayLike,
    n_pivots: int,
    kernel: Callable[ArrayLike, ArrayLike, Dict[str, Parameter]],
    kernel_params: Dict[str, Parameter],
) -> Tuple[Array, Array]:
    """Randomly Pivoted Cholesky decomposition

    This functions performs the Randomly Pivoted Cholesky decomposition
    (RPCholesky). The implementation follows Algorithm 2.1 of Ref. [1].
    The RPCholesky builds iteratively a Nystrom approximation for the kernel
    K:

            K_nystr = K(:, S) K(S, S)^-1 K(S, :)

    where K(:, S) is the N x S matrix of kernel evaluations between the
    N points and a subset of S points (pivots), and K(S, S) is the
    corresponding S x S matrix.
    The set of pivot is extracted randomly in RPCholesky, according to a
    probability distribution that weights more large diagonal entries, but
    puts a non-zero probability also to smaller elements. Compared to
    diagonal pivoting, this selection method is more robust to outliers [1].

    Args:
        key: JAX PRNG key
        x: input point, (n_samples, n_features)
        n_pivots: number of pivots to use
        kernel: GPX kernel
                must be called with signature kernel(x, x, kernel_params)
        kernel_params: kernel parameters

    Returns:
        fmat: factor matrix, shape (n_samples, n_pivots)
              The Nystrom approximation for the kernel is built as:

                  K_nystr = fmat @ fmat.T

        pivots: sampled pivots, shape (n_pivots,)

    References:
        [1] Yifan Chen, Ethan N. Epperly, Joel A. Tropp, and Robert J. Webber,
            Randomly pivoted Cholesky: Practical approximation of a kernel matrix
            with few entry evaluations (2023) arXiv:2207 (accessed 18 Oct 2023).
    """

    # x is (number of points, number of features)
    if kernel.nperms is not None:
        nperms = kernel.nperms
        np, nf = x.shape
        n = int(np / nperms)
        x = x.reshape(n, nperms, nf)
    else:
        n, _ = x.shape

    # factor matrix (F)
    fmat = jnp.zeros((n, n_pivots))

    # list of pivot indices (S)
    pivots = jnp.zeros((n_pivots,), dtype=int)

    # compute the diagonal
    def diagonal_wrapper(init, x):
        # ensure that each point is (1,)
        if kernel.nperms is None:
            x = jnp.expand_dims(x, axis=0)
        return init, kernel(x, x, kernel_params).squeeze()

    _, diag = jax.lax.scan(diagonal_wrapper, init=0, xs=x)

    # iteratively build the Nystrom approximation
    def fori_wrapper(i, val):
        fmat, diag, pivots, key = val

        # sample s ~ d / sum(d)
        prob = diag / jnp.sum(diag)
        key, subkey = random.split(key)
        s = jax.random.choice(key, jnp.arange(prob.shape[0]), p=prob, shape=())

        # update the pivots
        pivots = pivots.at[i].set(s)

        # compute the schur complement g
        # note: we are unable to select only a few columns of F due to
        #       the compilation within lax.fori_loop
        if kernel.nperms is None:
            xs = jnp.expand_dims(x[s], axis=0)
            g = kernel(xs, x, kernel_params) - jnp.dot(fmat, fmat[s].T)
        else:
            xs = x[s]
            g = kernel(xs, x.reshape(np, nf), kernel_params) - jnp.dot(fmat, fmat[s].T)
        g = g.squeeze()

        # update the i-th column of the factor matrix
        # note: 1e-6 is a jitter factor that ensures we are not dividing by 0
        fmat = fmat.at[:, i].set(g / (g[s] ** 0.5 + 1e-6))

        # update the diagonal
        diag = diag - fmat[:, i] ** 2

        return (fmat, diag, pivots, key)

    res = jax.lax.fori_loop(
        0, n_pivots, fori_wrapper, init_val=(fmat, diag, pivots, key)
    )

    fmat, _, pivots, _ = res

    return fmat, pivots


@partial(jit, static_argnums=(3, 4))
def rpcholesky_derivs(
    key: KeyArray,
    x: ArrayLike,
    jacobian: ArrayLike,
    n_pivots: int,
    kernel: Callable[ArrayLike, ArrayLike, Dict[str, Parameter]],
    kernel_params: Dict[str, Parameter],
) -> Tuple[Array, Array]:
    """Randomly Pivoted Cholesky decomposition

    This functions performs the Randomly Pivoted Cholesky decomposition
    (RPCholesky). The implementation follows Algorithm 2.1 of Ref. [1].
    The RPCholesky builds iteratively a Nystrom approximation for the kernel
    hessian:

            K_nystr = K(:, S) K(S, S)^-1 K(S, :)

    where K(:, S) is the N x S matrix of kernel evaluations between the
    N points and a subset of S points (pivots), and K(S, S) is the
    corresponding S x S matrix.
    The set of pivot is extracted randomly in RPCholesky, according to a
    probability distribution that weights more large diagonal entries, but
    puts a non-zero probability also to smaller elements. Compared to
    diagonal pivoting, this selection method is more robust to outliers [1].

    Args:
        key: JAX PRNG key
        x: input point, (n_samples, n_features)
        jacobian: jacobian of x, (n_samples,n_features,n_jac)
        n_pivots: number of pivots to use
        kernel: GPX kernel
                must be called with signature
                kernel.d01kj(x, x, kernel_params,jacobian,jacobian)
        kernel_params: kernel parameters

    Returns:
        fmat: factor matrix, shape (n_samples*n_jac, n_pivots*n_jac)
              The Nystrom approximation for the kernel is built as:

                  K_nystr = fmat @ fmat.T

        pivots: sampled pivots, shape (n_pivots,)

    References:
        [1] Yifan Chen, Ethan N. Epperly, Joel A. Tropp, and Robert J. Webber,
            Randomly pivoted Cholesky: Practical approximation of a kernel matrix
            with few entry evaluations (2023) arXiv:2207 (accessed 18 Oct 2023).
    """

    # x is (number of points, number of features)
    if kernel.nperms is not None:
        nperms = kernel.nperms
        np, nf, jv = jacobian.shape
        n = int(np / nperms)
        x = x.reshape(n, nperms, nf)
        jacobian = jacobian.reshape(n, nperms, nf, jv)
    else:
        n, _ = x.shape
        _, _, jv = jacobian.shape

    # factor matrix (F)
    fmat = jnp.zeros((n * jv, n_pivots))

    # list of pivot indices (S)
    pivots = jnp.zeros((n_pivots,), dtype=int)

    # compute the diagonal
    def diagonal_wrapper(init, xj):
        # ensure that each point is (1,)
        x, jacobian = xj
        if kernel.nperms is None:
            x = jnp.expand_dims(x, axis=0)
            jacobian = jnp.expand_dims(jacobian, axis=0)
        return init, kernel.d01kj(x, x, kernel_params, jacobian, jacobian)

    _, diag = jax.lax.scan(diagonal_wrapper, init=0, xs=(x, jacobian))
    diag = jnp.diagonal(diag, axis1=1, axis2=2).reshape(-1)

    # iteratively build the Nystrom approximation
    def fori_wrapper(i, val):
        fmat, diag, pivots, key = val

        # sample s ~ d / sum(d)
        prob = diag / jnp.sum(diag)
        key, subkey = random.split(key)
        s = jax.random.choice(key, jnp.arange(prob.shape[0]), p=prob, shape=())

        # update the pivots
        pivots = pivots.at[i].set(s)

        # compute the schur complement g
        # note: we are unable to select only a few columns of F due to
        #       the compilation within lax.fori_loop
        index1 = (s / jv).astype(int)
        index2 = jnp.mod(s, jv)
        if kernel.nperms is None:
            xs = jnp.expand_dims(x[index1], axis=0)
            js = jnp.expand_dims(jacobian[index1], axis=0)
            g = kernel.d01kj(xs, x, kernel_params, js, jacobian)[index2] - jnp.dot(
                fmat, fmat[s].T
            )
        else:
            xs = x[index1]
            js = jacobian[index1]
            g = kernel.d01kj(
                xs, x.reshape(np, nf), kernel_params, js, jacobian.reshape(np, nf, jv)
            )[index2] - jnp.dot(fmat, fmat[s].T)
        # g = g.squeeze()

        # update the i-th column of the factor matrix
        # note: 1e-6 is a jitter factor that ensures we are not dividing by 0
        fmat = fmat.at[:, i].set(g / (g[s] ** 0.5 + 1e-6))

        # update the diagonal
        diag = diag - fmat[:, i] ** 2

        return (fmat, diag, pivots, key)

    res = jax.lax.fori_loop(
        0, n_pivots, fori_wrapper, init_val=(fmat, diag, pivots, key)
    )

    fmat, _, pivots, _ = res

    return fmat, pivots
