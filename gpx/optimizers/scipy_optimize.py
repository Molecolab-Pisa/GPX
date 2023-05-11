from __future__ import annotations

from typing import Callable, Tuple

from jax import grad, jit
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_unflatten
from jax.typing import ArrayLike
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult

from ..parameters import ModelState

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
        # this ensures that gradients are stopped for parameters with
        # trainable = False.
        params = unravel_forward(xt)
        state = state.update(dict(params=params))
        return loss_fn(state, x, y)

    grad_loss = jit(grad(loss))
    optres = minimize(loss, x0=x0, args=(state), method="L-BFGS-B", jac=grad_loss)

    params = unravel_forward(optres.x)
    state = state.update(dict(params=params))

    return state, optres
