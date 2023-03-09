from __future__ import annotations
from typing import Tuple, Callable

from .parameters.model_state import ModelState

from jax.tree_util import tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
from jax import grad, jit
import jax.numpy as jnp

from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult


# ============================================================================
# Scipy Optimizer interface
# ============================================================================


def scipy_minimize(
    state: ModelState, x: jnp.ndarray, y: jnp.ndarray, loss_fn: Callable
) -> Tuple[ModelState, OptimizeResult]:

    fwd_fns = state.params_forward_transforms
    bwd_fns = state.params_backward_transforms

    def forward(xt):
        return [fwd(x) for fwd, x in zip(fwd_fns, xt)]

    def backward(xt):
        return [bwd(x) for bwd, x in zip(bwd_fns, xt)]

    def ravel_backward(params):
        x, tdef = tree_flatten(params)
        x = backward(x)
        x, unravel_fn = ravel_pytree(x)
        return x, tdef, unravel_fn

    x0, tdef, unravel_fn = ravel_backward(state.params)

    def unravel_forward(x):
        x = unravel_fn(x)
        x = forward(x)
        params = tree_unflatten(tdef, x)
        return params

    def loss(xt, state):
        # important: here we first reconstruct the model state with the
        # updated parameters before feeding it to the loss (lml).
        # this ensures that gradients are stopped for parameter with
        # trainable = False.
        params = unravel_forward(xt)
        state = state.update(dict(params=params))
        return loss_fn(state, x, y)

    grad_loss = jit(grad(loss))
    optres = minimize(loss, x0=x0, args=(state), method="L-BFGS-B", jac=grad_loss)

    params = unravel_forward(optres.x)
    state = state.update(dict(params=params))

    return state, optres
