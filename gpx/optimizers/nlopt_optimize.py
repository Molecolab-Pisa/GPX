from __future__ import annotations

import warnings
from typing import Callable, Tuple

import numpy as np
from jax import Array, grad, jit
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_unflatten
from jax.typing import ArrayLike

from ..parameters import ModelState

try:
    import nlopt
except ImportError:
    warnings.warn(
        "nlopt is not installed. Interface to NLopt optimizers is not available.",
        stacklevel=2,
    )


# TODO: This bridge is practically the same code used in our scipy
#       interface. Incorporate the two.
def _nlopt_bridge(
    state: ModelState, x: ArrayLike, y: ArrayLike, loss_fn: Callable
) -> Tuple[Array, Callable, Callable, Callable, Callable, Callable]:
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

    def loss(xt):
        # important: here we first reconstruct the model state with the
        # updated parameters before feeding it to the loss (lml).
        # this ensures that gradients are stopped for parameters with
        # trainable = False.
        params = unravel_forward(xt)
        ustate = state.update(dict(params=params))
        return loss_fn(ustate, x, y)

    grad_loss = jit(grad(loss))

    def nlopt_loss(x, grad):
        if grad.size > 0:
            # gradients from JAX
            _grad = grad_loss(x)
            grad[:] = np.array(_grad)
        return np.array(loss(x)) * 1.0  # the 1.0 is a hack to a silent issue

    return x0, ravel_backward, unravel_forward, loss, grad_loss, nlopt_loss


class NLoptWrapper:
    def __init__(
        self, state: ModelState, x: ArrayLike, y: ArrayLike, loss_fn: Callable, opt: int
    ) -> None:
        # store the state because it's needed later to
        # reconstruct the parameters
        self._state = state
        self.loss_fn = loss_fn

        # make the `nlopt_loss` function that exposes the JAX loss
        # to NLopt
        bridge = _nlopt_bridge(state=state, x=x, y=y, loss_fn=loss_fn)
        x0, ravel_backward, unravel_forward, loss, grad_loss, nlopt_loss = bridge

        # store the starting point, the functions to unbound/bound
        # the parameters, and the various losses in unbound space
        self.x0 = x0
        self.ravel_backward = ravel_backward
        self.unravel_forward = unravel_forward
        self.loss = loss
        self.grad_loss = grad_loss
        self.nlopt_loss = nlopt_loss

        # setup the NLopt optimizer
        self.opt = nlopt.opt(opt, len(x0))
        self.opt.set_min_objective(nlopt_loss)

    def optimize(self) -> ModelState:
        xt = self.opt.optimize(self.x0)
        params = self.unravel_forward(xt)
        state = self._state.update(dict(params=params))
        return state
