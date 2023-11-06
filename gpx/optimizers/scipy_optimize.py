from __future__ import annotations

from typing import Callable, Tuple

from jax import grad, jit
from jax.typing import ArrayLike
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from ..parameters import ModelState
from .utils import ravel_backward_trainables, unravel_forward_trainables

# ============================================================================
# Scipy Optimizer interface
# ============================================================================


def scipy_minimize(
    state: ModelState, x: ArrayLike, y: ArrayLike, loss_fn: Callable
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

    grad_loss = jit(grad(loss))

    optres = minimize(loss, x0=x0, method="L-BFGS-B", jac=grad_loss)

    params = unravel_forward(optres.x)
    state = state.update(dict(params=params))

    return state, optres


def scipy_minimize_derivs(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    loss_fn: Callable,
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
        return loss_fn(ustate, x, y, jacobian)

    grad_loss = jit(grad(loss))

    optres = minimize(loss, x0=x0, method="L-BFGS-B", jac=grad_loss)

    params = unravel_forward(optres.x)
    state = state.update(dict(params=params))

    return state, optres
