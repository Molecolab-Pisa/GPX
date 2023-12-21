from __future__ import annotations

import os
from functools import partial
from typing import Callable, Optional, Tuple

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

    @jit
    def loss(xt):
        # go in bound space and reconstruct params
        params = unravel_forward(xt)
        ustate = state.update(dict(params=params))
        return loss_fn(ustate, x, y)

    grad_loss = jit(grad(loss))

    optres = minimize(loss, x0=x0, method="L-BFGS-B", jac=grad_loss, callback=callback)

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


# ============================================================================
# Callbacks
# ============================================================================

# we use a dictionary storing a separate counter for each optimization
# (we don't want the counter to be overridden)
_STATE_CHECKPOINT_COUNTER = {}


def state_checkpointer(
    state: ModelState, chk_file: Optional[str] = "optim_chk.npz"
) -> Callable[ArrayLike]:
    """state checkpointer callable

    Generates a function that can be passed as `callback` to scipy_minimize.
    This callable reconstructs the ModelState and saves it to a .npz file.
    You can then load back the model parameters into a model with `model.load`.

    Args:
        state: model state.
        chk_file: name of the checkpoint file. Note that a checkpoint will be
                  saved for each step of scipy_minimize with a different postfix.
                  E.g., if chk_file='test.npz', you will obtain 'test.000.npz'
                  and so on.
    Returns:
        callback: function that can be passed as `callback` argument to
                  scipy_minimize
    """
    # use the hash code of state as key for the checkpoint counter
    # WARNING: if you start from the exact same state you override
    #          the counter

    global _STATE_CHECKPOINT_COUNTER

    # create counter
    hash_code = hash(state)
    _STATE_CHECKPOINT_COUNTER[hash_code] = 0

    # get the fmt string in order to save the state with a counter
    name, ext = os.path.splitext(chk_file)
    chk_file = name + ".{:03d}" + ext

    def callback(x):
        global _STATE_CHECKPOINT_COUNTER
        counter = _STATE_CHECKPOINT_COUNTER[hash_code]

        _, tdef, unravel_fn = ravel_backward_trainables(state.params)
        unravel_forward = unravel_forward_trainables(unravel_fn, tdef, state.params)
        params = unravel_forward(x)
        ustate = state.update(dict(params=params))
        ustate.save(chk_file.format(counter))

        # update counter
        _STATE_CHECKPOINT_COUNTER[hash_code] += 1

    return callback
