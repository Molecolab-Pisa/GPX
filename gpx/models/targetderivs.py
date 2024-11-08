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

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array, grad, jit, lax, random, value_and_grad
from jax.scipy.sparse.linalg import cg
from jax.typing import ArrayLike
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from ..mean_functions import zero_mean
from ..operations import recover_first_axis, update_row_diagonal
from ..optimizers.utils import ravel_backward_trainables, unravel_forward_trainables
from ..parameters import ModelState, Parameter
from .utils import randomized_minimization_ol

ParameterDict = Dict[str, Parameter]
Kernel = Any
KeyArray = Array


def _Ax_lhs_fun(
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    noise: Optional[bool] = True,
) -> Callable[ArrayLike, Array]:

    kernel_params = params["kernel_params"]
    sigma_targets = params["sigma_targets"].value
    sigma_derivs = params["sigma_derivs"].value

    @jit
    def matvec(z):

        ns, _, jv = jacobian2.shape

        z1 = z[:ns]
        z2 = z[ns:]

        def update_row(carry, xjs):

            x1s, j1s = recover_first_axis(xjs)

            row11 = kernel(x1s, x2, kernel_params)
            if noise:
                jitter_noise1 = sigma_targets**2 + 1e-10
                row11 = update_row_diagonal(row11, carry, jitter_noise1)

            row12 = kernel.d1kj(x1s, x2, kernel_params, jacobian2)

            rowvec1 = jnp.dot(row11, z1) + jnp.dot(row12, z2)

            row21 = kernel.d0kj(x1s, x2, kernel_params, j1s)
            row22 = kernel.d01kj(x1s, x2, kernel_params, j1s, jacobian2)
            if noise:
                jitter_noise2 = sigma_derivs**2 + 1e-10
                row22 = update_row_diagonal(row22, carry, jitter_noise2)

            rowvec2 = jnp.dot(row21, z1) + jnp.dot(row22, z2)

            rowvec = jnp.concatenate((rowvec1, rowvec2), axis=0)

            carry = carry + 1

            return carry, rowvec

        _, res = lax.scan(update_row, 0, (x1, jacobian1))
        res = jnp.concatenate(res, axis=0)

        ns, _, jv = jacobian1.shape

        res1 = res[jnp.arange(0, ns + ns * jv, jv + 1)]
        res2 = jnp.delete(res, np.arange(0, ns + ns * jv, jv + 1), axis=0)

        return jnp.concatenate((res1, res2))

    return matvec


@partial(jit, static_argnums=(3, 4))
def rpcholesky_TD(
    key: KeyArray,
    x: ArrayLike,
    jacobian: ArrayLike,
    n_pivots: int,
    kernel: Kernel,
    kernel_params: Dict[str, Parameter],
) -> Tuple[Array, Array]:

    # x is (number of points, number of features)
    n, _ = x.shape
    _, _, jv = jacobian.shape

    # factor matrix (F)
    fmat = jnp.zeros((n + n * jv, n_pivots))

    # list of pivot indices (S)
    pivots = jnp.zeros((n_pivots,), dtype=int)

    # compute the diagonal
    def diagonal_wrapper(init, xj):
        # ensure that each point is (1,)
        x, jacobian = xj
        x = jnp.expand_dims(x, axis=0)
        jacobian = jnp.expand_dims(jacobian, axis=0)

        return init, (
            kernel(x, x, kernel_params).squeeze(),
            kernel.d01kj(x, x, kernel_params, jacobian, jacobian),
        )

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

        def true_fun():
            k = kernel(jnp.expand_dims(x[s], axis=0), x, kernel_params).squeeze()
            d1kj = kernel.d1kj(
                jnp.expand_dims(x[s], axis=0), x, kernel_params, jacobian
            ).squeeze()
            return jnp.concatenate((k, d1kj), axis=0) - jnp.dot(fmat, fmat[s].T)

        def false_fun():
            index1 = ((s - n) / jv).astype(int)
            index2 = jnp.mod((s - n), jv)
            xs = jnp.expand_dims(x[index1], axis=0)
            js = jnp.expand_dims(jacobian[index1], axis=0)
            d0kj = kernel.d0kj(xs, x, kernel_params, js)[index2]
            d01kj = kernel.d01kj(xs, x, kernel_params, js, jacobian)[index2]
            return jnp.concatenate((d0kj, d01kj), axis=0) - jnp.dot(fmat, fmat[s].T)

        g = jnp.where(s < n, true_fun(), false_fun())

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
    x1: ArrayLike,
    jacobian1: ArrayLike,
    x2: ArrayLike,
    jacobian2: ArrayLike,
    params: ParameterDict,
    kernel: Kernel,
    n_pivots: int,
    key: KeyArray,
) -> Callable[ArrayLike, Array]:

    kernel_params = params["kernel_params"]
    sigma_targets = params["sigma_targets"].value

    fmat, _ = rpcholesky_TD(
        key=key,
        x=x1,
        jacobian=jacobian1,
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


def scipy_minimize_TD(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    y_derivs: ArrayLike,
    jacobian: ArrayLike,
    loss_fn: Callable,
    callback: Optional[Callable] = None,
) -> Tuple[ModelState, OptimizeResult]:

    # x0: flattened trainables (1D) in unbound space
    # tdef: definition of trainables tree (non-trainables are None)
    # unravel_fn: callable to unflatten x0
    x0, tdef, unravel_fn = ravel_backward_trainables(state.params)

    # function to unravel and unflatten trainables and go in bound space
    unravel_forward = unravel_forward_trainables(unravel_fn, tdef, state.params)

    def loss(xt):
        # go in bound space and reconstruct params
        params = unravel_forward(xt)
        ustate = state.update(dict(params=params))
        return loss_fn(state=ustate, x=x, y=y, y_derivs=y_derivs, jacobian=jacobian)

    loss_and_grad = jit(value_and_grad(loss))
    jax.debug.print("{z}", z=jit(grad(loss))(x0))

    optres = minimize(
        loss_and_grad, x0=x0, method="L-BFGS-B", jac=True, callback=callback
    )

    params = unravel_forward(optres.x)
    state = state.update(dict(params=params))

    return state, optres


class TargetsDerivs:
    def __init__(
        self,
        kernel: Kernel,
        mean_function: Callable = zero_mean,
        kernel_params: Dict[str, Parameter] = None,
        sigma_targets: Parameter = None,
        sigma_derivs: Parameter = None,
    ) -> None:

        params = {
            "kernel_params": kernel_params,
            "sigma_targets": sigma_targets,
            "sigma_derivs": sigma_derivs,
        }
        opt = {
            "x_train": None,
            "jacobian_train": None,
            "jaccoef": None,
            "y_train": None,
            "y_derivs_train": None,
            "is_fitted": False,
            "c": None,
            "c_targets": None,
            "mu": None,
        }

        self.state = ModelState(kernel, mean_function, params, **opt)

    @partial(jit, static_argnums=(0, 6, 7))
    def _lml_dense(
        self,
        params: ParameterDict,
        x: ArrayLike,
        jacobian: ArrayLike,
        y: ArrayLike,
        y_derivs: ArrayLike,
        kernel: Kernel,
        mean_function: Callable[ArrayLike, Array],
    ) -> Array:
        m = y.shape[0]
        kernel_params = params["kernel_params"]
        sigma_targets = params["sigma_targets"].value
        sigma_derivs = params["sigma_derivs"].value

        mu = mean_function(y)
        y = y - mu
        y = y.reshape(-1, 1)
        y_derivs = y_derivs.reshape(-1, 1)
        y_m = jnp.concatenate((y, y_derivs))

        # build kernel with target and derivatives
        K = kernel(x1=x, x2=x, params=kernel_params)
        K = K + sigma_targets**2 * jnp.eye(K.shape[0])

        D01kj = kernel.d01kj(
            x1=x,
            jacobian1=jacobian,
            x2=x,
            jacobian2=jacobian,
            params=kernel_params,
        )
        D01kj = D01kj + sigma_derivs**2 * jnp.eye(D01kj.shape[0])

        D0kj = kernel.d0kj(
            x1=x,
            x2=x,
            params=kernel_params,
            jacobian=jacobian,
        )

        C_mm = jnp.concatenate(
            (
                jnp.concatenate((K, D0kj.T), axis=1),
                jnp.concatenate((D0kj, D01kj), axis=1),
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

    def log_marginal_likelihood(
        self,
        state: ModelState,
        x: ArrayLike,
        jacobian: ArrayLike,
        y: ArrayLike,
        y_derivs: ArrayLike,
    ) -> Array:
        return self._lml_dense(
            params=state.params,
            x=x,
            jacobian=jacobian,
            y=y,
            y_derivs=y_derivs,
            kernel=state.kernel,
            mean_function=state.mean_function,
        )

    def neg_log_marginal_likelihood(
        self,
        state: ModelState,
        x: ArrayLike,
        jacobian: ArrayLike,
        y: ArrayLike,
        y_derivs: ArrayLike,
    ) -> Array:
        return -self.log_marginal_likelihood(
            state=state,
            x=x,
            jacobian=jacobian,
            y=y,
            y_derivs=y_derivs,
        )

    @partial(jit, static_argnums=(0, 6, 7))
    def _fit_dense(
        self,
        params: ParameterDict,
        x: ArrayLike,
        jacobian: ArrayLike,
        y: ArrayLike,
        y_derivs: ArrayLike,
        kernel: Kernel,
        mean_function: Callable[ArrayLike, Array],
    ) -> Tuple[Array, Array]:

        kernel_params = params["kernel_params"]
        sigma_targets = params["sigma_targets"].value
        sigma_derivs = params["sigma_derivs"].value

        mu = mean_function(y)
        y = y - mu
        y = y.reshape(-1, 1)
        y_derivs = y_derivs.reshape(-1, 1)
        y_m = jnp.concatenate((y, y_derivs))

        # build kernel with target and derivatives
        K = kernel(x1=x, x2=x, params=kernel_params)
        K = K + (sigma_targets**2 + 1e-10) * jnp.eye(K.shape[0])

        D01kj = kernel.d01kj(
            x1=x,
            jacobian1=jacobian,
            x2=x,
            jacobian2=jacobian,
            params=kernel_params,
        )
        D01kj = D01kj + (sigma_derivs**2 + 1e-10) * jnp.eye(D01kj.shape[0])

        D0kj = kernel.d0kj(
            x1=x,
            x2=x,
            params=kernel_params,
            jacobian=jacobian,
        )

        C_mm = jnp.concatenate(
            (
                jnp.concatenate((K, D0kj.T), axis=1),
                jnp.concatenate((D0kj, D01kj), axis=1),
            ),
            axis=0,
        )
        c = jnp.linalg.solve(C_mm, y_m)
        return c, mu

    @partial(jit, static_argnums=(0, 6, 7, 8))
    def _fit_iter(
        self,
        params: ParameterDict,
        x: ArrayLike,
        jacobian: ArrayLike,
        y: ArrayLike,
        y_derivs: ArrayLike,
        kernel: Kernel,
        mean_function: Callable[ArrayLike, Array],
        n_pivots: int,
        key_precond: KeyArray,
    ) -> Tuple[Array, Array]:
        # calculate mean and flatten y_derivs
        mu = mean_function(y)
        y = y - mu
        y = y.reshape(-1, 1)
        y_derivs = y_derivs.reshape(-1, 1)
        y_m = jnp.concatenate((y, y_derivs))

        matvec = _Ax_lhs_fun(
            x1=x,
            jacobian1=jacobian,
            x2=x,
            jacobian2=jacobian,
            params=params,
            kernel=kernel,
            noise=True,
        )

        precond = _precond_rpcholesky_TD(
            x1=x,
            jacobian1=jacobian,
            x2=x,
            jacobian2=jacobian,
            params=params,
            kernel=kernel,
            n_pivots=n_pivots,
            key=key_precond,
        )

        c, _ = cg(matvec, y_m, M=precond, atol=1e-7)

        return c, mu

    def fit(
        self,
        x: ArrayLike,
        jacobian: ArrayLike,
        y: ArrayLike,
        y_derivs: ArrayLike,
        iterative: Optional[bool] = False,
        key: Optional[KeyArray] = None,
        num_restarts: Optional[int] = 0,
        minimize: Optional[bool] = True,
        return_history: Optional[bool] = False,
        n_pivots: Optional[int] = None,
        key_precond: Optional[KeyArray] = None,
    ) -> ModelState:

        if minimize:
            minimization_function = scipy_minimize_TD
            self.state, optres, *history = randomized_minimization_ol(
                key=key,
                state=self.state,
                x=x,
                y=y,
                y_derivs=y_derivs,
                jacobian=jacobian,
                loss_fn=self.neg_log_marginal_likelihood,
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
            partial(self._fit_iter, n_pivots=n_pivots, key_precond=key_precond)
            if iterative
            else self._fit_dense
        )

        c, mu = fit_func(
            params=self.state.params,
            x=x,
            jacobian=jacobian,
            y=y,
            y_derivs=y_derivs,
            kernel=self.state.kernel,
            mean_function=self.state.mean_function,
        )
        ns, _, nv = jacobian.shape

        c_targets = c[:ns].reshape(-1)
        c_derivs = c[ns : ns + nv * ns]
        jaccoef = jnp.einsum("sv,sfv->sf", c_derivs.reshape(ns, nv), jacobian)

        self.state = self.state.update(
            dict(
                x_train=x,
                jacobian_train=jacobian,
                jaccoef=jaccoef,
                y_train=y,
                y_derivs_train=y_derivs,
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

    @partial(jit, static_argnums=(0, 8))
    def _predict_dense(
        self,
        params: ParameterDict,
        x_train: ArrayLike,
        jacobian_train: ArrayLike,
        x: ArrayLike,
        jacobian: ArrayLike,
        c: ArrayLike,
        mu: ArrayLike,
        kernel: Kernel,
    ) -> Union[Array, Tuple[Array, Array]]:

        kernel_params = params["kernel_params"]

        K = kernel(x1=x_train, x2=x, params=kernel_params)

        D01kj = kernel.d01kj(
            x1=x_train,
            jacobian1=jacobian_train,
            x2=x,
            jacobian2=jacobian,
            params=kernel_params,
        )

        D0kj = kernel.d0kj(
            x1=x_train,
            x2=x,
            params=kernel_params,
            jacobian=jacobian_train,
        )

        D1kj = kernel.d1kj(
            x1=x_train,
            x2=x,
            params=kernel_params,
            jacobian=jacobian,
        )

        K_mn = jnp.concatenate(
            (
                jnp.concatenate((K, D1kj), axis=1),
                jnp.concatenate((D0kj, D01kj), axis=1),
            ),
            axis=0,
        )

        pred = jnp.dot(K_mn.T, c)

        ns, _, nv = jacobian.shape

        pred = pred.at[:ns, :].add(mu)
        y_pred = pred[:ns]
        y_derivs_pred = pred[ns : ns + nv * ns].reshape(ns, -1)

        return y_pred, y_derivs_pred

    @partial(
        jit,
        static_argnums=(
            0,
            8,
        ),
    )
    def _predict_iter(
        self,
        params: ParameterDict,
        x_train: ArrayLike,
        jacobian_train: ArrayLike,
        x: ArrayLike,
        jacobian: ArrayLike,
        c: ArrayLike,
        mu: ArrayLike,
        kernel: Kernel,
    ):
        """predicts derivative values with GPR

        Predicts the derivative values with GPR.
        The contraction with the linear coefficients is performed
        iteratively.

        μ_n = K_nm (K_mm + σ²)⁻¹(y - μ)

        where K = ∂₁∂₂K
        """
        matvec = _Ax_lhs_fun(
            x1=x,
            jacobian1=jacobian,
            x2=x_train,
            jacobian2=jacobian_train,
            params=params,
            kernel=kernel,
            noise=False,
        )
        pred = matvec(c)
        # recover the right shape
        ns, _ = x.shape

        pred = pred.at[:ns, :].add(mu)
        y_pred = pred[:ns]
        y_derivs_pred = pred[ns:].reshape(ns, -1)
        return y_pred, y_derivs_pred

    def predict(
        self,
        x: ArrayLike,
        jacobian: ArrayLike,
        iterative: Optional[bool] = False,
    ):

        if not self.state.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Run `fit` to fit the model before prediction."
            )
        if iterative:
            return self._predict_iter(
                params=self.state.params,
                x_train=self.state.x_train,
                jacobian_train=self.state.jacobian_train,
                x=x,
                jacobian=jacobian,
                c=self.state.c,
                mu=self.state.mu,
                kernel=self.state.kernel,
            )

        return self._predict_dense(
            params=self.state.params,
            x_train=self.state.x_train,
            jacobian_train=self.state.jacobian_train,
            x=x,
            jacobian=jacobian,
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
