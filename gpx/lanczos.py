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
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.typing import ArrayLike

KeyArray = Array


def orthogonalize(v: ArrayLike, vecs: ArrayLike) -> Array:
    "orthogonalize v w.r.t all the vectors in vecs"

    def update_fun(origv, refv):
        origv = origv - jnp.dot(origv, refv) * refv
        return origv, None

    ortho_vec, _ = jax.lax.scan(update_fun, v, vecs.T)
    return ortho_vec


def orthonormalize(v: ArrayLike, vecs: ArrayLike) -> Array:
    "orthonormalize v w.r.t. all the vectors in vecs"
    ortho_vec = orthogonalize(v, vecs)
    return ortho_vec / jnp.linalg.norm(ortho_vec)


@partial(jit, static_argnums=(0, 1))
def lanczos_tridiagonal(
    matvec: Callable[ArrayLike, Array], m: int, v1: ArrayLike, key: KeyArray
) -> Tuple[Array, Array]:
    """lanczos tridiagonalization

    Perform a Lanczos tridiagonalization of a n x n symmetric matrix A.
    The algorithm works without instantiating A: all is required is a `matvec`
    function matvec(z) computing the matrix-vector product A∙z.
    The Lanczos tridiagonalization is performed in `m` iterations, and outputs
    a m x m tridiagonal matrix T together with the Lanczos vectors `vecs` of
    dimension n x m.
    This implementation follows the one advocated by Paige [1, 2], with a further
    reorthogonalization step to improve its numerical stability.

    Args:
        matvec: function computing the matrix-vector product A∙z
        m: number of Lanczos iteration to compute the tridiagonal matrix T
        v1: initial random vector. This vector *must* be normalized, and be 1D
        key: random key used to generate new random vectors if needed.
    Returns:
        vecs: matrix of random vectors vj, (n, m)
        tridiag: tridiagonal matrix T, (m, m)
    References:
        [1] Jane K. Cullum, Ralph A. Willoughby
            Lanczos Algorithms for Large Symmetric Eigenvalue Problems
            https://epubs.siam.org/doi/epdf/10.1137/1.9780898719192.ch2
        [2] https://en.wikipedia.org/wiki/Lanczos_algorithm
    """
    _LANCZOS_BETA_THRESHOLD = 1e-12

    # initialization phase
    # matrix V (vecs)
    vecs = jnp.zeros((v1.shape[0], m), dtype=v1.dtype)
    vecs = vecs.at[:, 0].set(v1)
    # tridiagonal matrix T
    tridiag = jnp.zeros((m, m), dtype=v1.dtype)

    # first step of lanczos
    w = matvec(v1)
    alpha = jnp.dot(w, v1)
    w = w - alpha * v1
    beta = jnp.linalg.norm(w)

    # update tridiag with alpha_1
    tridiag = tridiag.at[0, 0].set(alpha)

    def gen_random_orthonormal(key, v1, vecs):
        randv = random.normal(key, shape=v1.shape, dtype=v1.dtype)
        randv = orthonormalize(randv, vecs)
        return randv

    def body_fun(j, val):
        # extract val
        beta, w, tridiag, vecs, key = val

        # sample a new vector
        subkey, key = random.split(key)
        v = jnp.where(
            beta > _LANCZOS_BETA_THRESHOLD,
            w / beta,
            gen_random_orthonormal(subkey, v1, vecs),
        )

        # update V
        vecs = vecs.at[:, j + 1].set(v)

        # generate the new w
        # the orthogonalization step improves the stability
        # of the algorithm
        w = matvec(v)
        alpha = jnp.dot(w, v)
        w = w - alpha * v - beta * vecs[:, j]
        w = orthogonalize(w, vecs)

        # update tridiag
        tridiag = tridiag.at[j + 1, j + 1].set(alpha)
        tridiag = tridiag.at[j, j + 1].set(beta)
        tridiag = tridiag.at[j + 1, j].set(beta)

        beta = jnp.linalg.norm(w)

        return beta, w, tridiag, vecs, key

    init_val = (beta, w, tridiag, vecs, key)
    _, _, tridiag, vecs, _ = jax.lax.fori_loop(0, m - 1, body_fun, init_val)

    return tridiag, vecs


@partial(jit, static_argnums=(0, 1, 2, 3))
def lanczos_logdet(
    matvec: Callable[ArrayLike, Array],
    num_evals: int,
    dim_mat: int,
    num_lanczos: int,
    key: KeyArray,
) -> Array:
    """approximate log(det(A)) = tr(log(A)) with stochastic trace estimation

    Approximate the log determinant of a matrix A, also expressed as the trace
    of the log(A), with stochastic trace estimation (Monte Carlo) and the Lanczos
    tridiagonalization of A [1, 2].

    Args:
        matvec: function computing the matrix-vector product A∙z
        m: number of Lanczos iteration to compute the tridiagonal matrix T
        v1: initial random vector. This vector *must* be normalized, and be 1D
        key: random key used to generate new random vectors if needed.
    Returns:
        vecs: matrix of random vectors vj, (n, m)
        tridiag: tridiagonal matrix T, (m, m)
    References:
        [1] Ubaru, S., Chen, J., Saad, Y.
            Fast Estimation of tr(F(A)) Via Stochastic Lanczos Quadrature
            https://doi.org/10.1137/16M1104974
        [2] Dong, K., Eriksson, D., Nickisch, H., Bindel, D., Wilson, A. G.
            Scalable Log Determinants for Gaussian Process Kernel Learning
            arXiv:1711.03481v1
    """
    # extract random vectors with rademacher distribution
    # also normalize them as required by `lanczos_tridiag`
    subkey, key = random.split(key)
    us = random.rademacher(subkey, shape=(dim_mat, num_evals))
    us = us / jnp.linalg.norm(us, axis=0)

    def update_fun(logdet, u):
        tridiag, _ = lanczos_tridiagonal(matvec=matvec, m=num_lanczos, v1=u, key=key)
        # maybe jax.scipy.linalg.eigh_tridiagonal is faster?
        evals, evecs = jnp.linalg.eigh(tridiag)
        # force eigenvalues to be positive
        evals = jnp.maximum(evals, 1e-36)
        tauk2 = jnp.square(evecs[0, :])
        logdet = logdet + jnp.sum(tauk2 * jnp.log(evals))
        return logdet, None

    logdet, _ = jax.lax.scan(update_fun, 0.0, (us.T))
    return (dim_mat / num_evals) * logdet
