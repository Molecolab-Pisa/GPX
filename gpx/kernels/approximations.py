from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, random
from jax._src import prng
from jax.typing import ArrayLike

Parameter = Any


@partial(jit, static_argnums=(2, 3))
def rpcholesky(
    key: prng.PRNGKeyArray,
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
    n, _ = x.shape

    # factor matrix (F)
    fmat = jnp.zeros((n, n_pivots))

    # list of pivot indices (S)
    pivots = jnp.zeros((n_pivots,), dtype=int)

    # compute the diagonal
    def diagonal_wrapper(init, x):
        # ensure that each point is (1,)
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
        g = kernel(jnp.expand_dims(x[s], axis=0), x, kernel_params) - jnp.dot(
            fmat, fmat[s].T
        )
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
