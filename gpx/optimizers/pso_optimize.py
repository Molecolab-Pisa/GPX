from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.typing import ArrayLike

from gpx.bijectors import Identity
from gpx.optimizers.scipy_optimize import scipy_minimize
from gpx.optimizers.utils import ravel_backward_trainables, unravel_forward_trainables
from gpx.parameters import ModelState
from gpx.parameters.parameter import is_parameter

try:
    import pso_jax
except ImportError:
    warnings.warn(
        """PSO-JAX is not installed. Interface to PSO-JAX optimizer is
         not available.""",
        stacklevel=2,
    )


def replace_bijector_with_identity(p):
    return p.update(dict(bijector=Identity()))


def pso_minimize(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    boundary: List,
    loss_fn: Callable,
    repeat: Optional[int] = 1,
    localopt: Optional[bool] = True,
    **kwargs,
) -> ModelState:
    """minimization of a loss function using PSO

    Performs the minimization of the `loss_fn` loss function using PSO.

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
        boundary: boundary conditions of the parameters to be optimized
        loss_fn: loss function with signature loss_fn(state, x, y),
                 returning a scalar value
       repeat: number of PSO restarts
       localopt: whether to do a local optimization using L-BFGS-B after
                 the PSO or not
    Returns:
        state: updated model state

    Note:
        The GPX bijectors are not used in PSO-JAX, but they are used at
        the end if minimization with L-BFGS-B is required using scipy.
    """

    # state with Identity bijector
    fake_params = tree_map(
        replace_bijector_with_identity, state.params, is_leaf=is_parameter
    )
    fake_state = state.update(dict(params=fake_params))

    # x0: flattened trainables (1D) in unbound space
    # tdef: definition of trainables tree (non-trainables are None)
    # unravel_fn: callable to unflatten x0

    x0, tdef, unravel_fn = ravel_backward_trainables(fake_state.params)

    # function to unravel and unflatten trainables and go in bound space
    unravel_forward = unravel_forward_trainables(unravel_fn, tdef, fake_state.params)

    # @jit
    def loss(xt):
        # go in bound space and reconstruct params
        params = unravel_forward(xt)
        ustate = fake_state.update(dict(params=params))
        return loss_fn(ustate, x, y)

    fn = jnp.inf
    for _ in range(repeat):
        trial_optres, trial_fn, history_fn, history_optres = pso_jax.PSO(
            loss, boundary, seed=_, **kwargs
        )
        if trial_fn < fn:
            fn = trial_fn
            optres = trial_optres

    params = unravel_forward(optres)
    fake_state = fake_state.update(dict(params=params))
    optim_params = tree_map(
        lambda op, fp: op.update(dict(value=fp.value)),
        state.params,
        fake_state.params,
        is_leaf=is_parameter,
    )
    state = state.update(dict(params=optim_params))

    if localopt:
        state, optres = scipy_minimize(state, x, y, loss_fn)

    return state
