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

import os
from functools import partial
from typing import Callable, Optional, Tuple

from jax import jit, value_and_grad
from jax.typing import ArrayLike
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from ..parameters import ModelState
from .utils import ravel_backward_trainables, unravel_forward_trainables

# ============================================================================
# Scipy Optimizer interface
# ============================================================================


def scipy_minimize(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    loss_fn: Callable,
    callback: Optional[Callable] = None,
) -> Tuple[ModelState, OptimizeResult]:
    """minimization of a loss function using SciPy's L-BFGS-B

    Performs the minimization of the `loss_fn` loss function using
    SciPy's L-BFGS-B optimizator.

    Args:
        state: model state. Should have an attribute `params`
               containing the parameters to be optimized, and
               attributes `params_forward_transforms` and
               `params_backward_transforms` storing the forward
               and backward function that constraining the values
               of the hyperparameters.
               The state is passed as an argument to the loss function.
        x: observations
        y: target values
        loss_fn: loss function with signature loss_fn(state, x, y),
                 returning a scalar value
    Returns:
        state: updated model state
        optres: optimization results, as output by the SciPy's optimizer
    """

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
        return loss_fn(ustate, x, y)

    loss_and_grad = jit(value_and_grad(loss))

    optres = minimize(
        loss_and_grad, x0=x0, method="L-BFGS-B", jac=True, callback=callback
    )

    params = unravel_forward(optres.x)
    state = state.update(dict(params=params))

    return state, optres


def scipy_minimize_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    loss_fn: Callable,
    callback: Optional[Callable] = None,
) -> Tuple[ModelState, OptimizeResult]:
    """minimization of a loss function using SciPy's L-BFGS-B

    Performs the minimization of the `loss_fn` loss function using
    SciPy's L-BFGS-B optimizator. Passes the jacobian to the loss
    function, useful to train on derivative values.

    Args:
        state: model state. Should have an attribute `params`
               containing the parameters to be optimized, and
               attributes `params_forward_transforms` and
               `params_backward_transforms` storing the forward
               and backward function that constraining the values
               of the hyperparameters.
               The state is passed as an argument to the loss function.
        x: observations
        y: target values
        jacobian: jacobian of x
        loss_fn: loss function with signature loss_fn(state, x, y),
                 returning a scalar value
    Returns:
        state: updated model state
        optres: optimization results, as output by the SciPy's optimizer
    """
    loss_fn = partial(loss_fn, jacobian=jacobian)
    return scipy_minimize(state=state, x=x, y=y, loss_fn=loss_fn, callback=callback)


def scipy_minimize_ol(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    y_derivs: ArrayLike,
    jacobian: ArrayLike,
    loss_fn: Callable,
    y_derivs2: ArrayLike = None,
    jacobian_2: ArrayLike = None,
    callback: Optional[Callable] = None,
) -> Tuple[ModelState, OptimizeResult]:
    """minimization of a loss function using SciPy's L-BFGS-B

    Performs the minimization of the `loss_fn` loss function using
    SciPy's L-BFGS-B optimizator. Passes the jacobian to the loss
    function, useful to train on derivative values.

    Args:
        state: model state. Should have an attribute `params`
               containing the parameters to be optimized, and
               attributes `params_forward_transforms` and
               `params_backward_transforms` storing the forward
               and backward function that constraining the values
               of the hyperparameters.
               The state is passed as an argument to the loss function.
        x: observations
        y: target values
        y_derivs: target derivatives
        jacobian: jacobian of x
        loss_fn: loss function with signature loss_fn(state, x, y),
                 returning a scalar value
    Returns:
        state: updated model state
        optres: optimization results, as output by the SciPy's optimizer
    """
    loss_fn = partial(loss_fn, y_derivs=y_derivs, y_derivs2=y_derivs2, jacobian=jacobian, jacobian_2=jacobian_2)
    return scipy_minimize(state=state, x=x, y=y, loss_fn=loss_fn, callback=callback)


# ============================================================================
# Callbacks
# ============================================================================


class StateCheckpointer:
    """state checkpointer callback

    This class can be passed as a 'callback' to scipy_minimize.
    When called, it reconstructs the ModelState and saves it to a .npz file.
    You can then load back the model parameters into a model with `model.load`
    to check the model along the optimization.
    You can also use the saved model to perform a restart if something goes
    wrong during a long optimization.
    """

    def __init__(
        self, state: ModelState, chk_file: Optional[str] = "optim_chk.npz"
    ) -> None:
        """
        Args:
            state: model state.
            chk_file: name of the checkpoint file. Note that a checkpoint will
                      be saved for each step of scipy_minimize with a
                      different postfix.
                      E.g., if chk_file='test.npz', you will obtain

                       - 'test.000.npz'
                       - 'test.001.npz'
                       - ...

                      and so on.
        """
        # store the state
        self.state = state
        # counter for how many times the callback is called
        self.counter = 0
        # create the formatted path for saving
        self.fmt_chk_file = chk_file
        # get the tree definition and the function to unravel the arrays
        _, tdef, unravel_fn = ravel_backward_trainables(state.params)
        # get the function to reconstruct the parameters from x
        self.unravel_forward = unravel_forward_trainables(
            unravel_fn, tdef, state.params
        )

    @property
    def fmt_chk_file(self):
        return self._fmt_chk_file

    @fmt_chk_file.setter
    def fmt_chk_file(self, value: str) -> None:
        name, ext = os.path.splitext(value)
        self._fmt_chk_file = name + ".{:03d}" + ext

    def __call__(self, x: ArrayLike) -> None:
        params = self.unravel_forward(x)
        ustate = self.state.update(dict(params=params))
        ustate.save(self.fmt_chk_file.format(self.counter))
        self.counter += 1
