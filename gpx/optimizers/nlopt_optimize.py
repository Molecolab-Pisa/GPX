from __future__ import annotations

import warnings
from typing import Callable, Tuple

import numpy as np
from jax import Array, grad, jit
from jax.typing import ArrayLike

from ..parameters import ModelState
from .utils import ravel_backward_trainables, unravel_forward_trainables

try:
    import nlopt
except ImportError:
    warnings.warn(
        "NLopt is not installed. Interface to NLopt optimizers is not available.",
        stacklevel=2,
    )


def _nlopt_bridge(
    state: ModelState, x: ArrayLike, y: ArrayLike, loss_fn: Callable
) -> Tuple[Array, Callable, Callable, Callable, Callable, Callable]:
    """wraps the loss function to work with NLopt

    Builds the bridge between NLopt and the GPX loss function.
    Creates the functions to transform the trainable parameters into
    unbound and bound space. Defines a loss function that takes into
    account the constraints, and creates a wrapper loss that is compatible
    with NLopt.
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

    def nlopt_loss(x, grad):
        if grad.size > 0:
            # gradients from JAX
            _grad = grad_loss(x)
            # in-place update
            grad[:] = np.array(_grad)
        return np.array(loss(x)) * 1.0  # the 1.0 is a hack to a silent issue

    return x0, ravel_backward_trainables, unravel_forward, loss, grad_loss, nlopt_loss


class NLoptWrapper:
    """interface to NLopt optimizers

    This is a really thin wrapper around NLopt optimizers.
    It automatically takes into account the transformations defined for
    the parameters (forward_transform, backward_transform), extracts
    the trainable-only parameters to be passed to the NLopt optimizer, and
    builds a wrapper around the loss_fn that is compatible with NLopt.
    Inside the wrapper, gradients are computed using JAX's automatic
    differentiation.

    It is designed to be used in a similar way to NLopt optimizer.
    E.g., from the NLopt tutorial

    >>> opt = nlopt.opt(nlopt.LD_MMA, 2)
    >>> opt.set_lower_bounds([-float('inf'), 0])
    >>> opt.set_min_objective(myfunc)
    >>> # [...]
    >>> opt.set_xtol_rel(1e-4)
    >>> x = opt.optimize([1.234, 5.678])
    >>> minf = opt.last_optimum_value()

    With this wrapper, you would do:

    >>> optim = NLoptWrapper(state, x=x, y=y, loss_fn=loss_fn, opt=nlopt.LD_MMA)
    >>> # you access the NLopt optimizer as `optim.opt`
    >>> optim.opt.set_lower_bounds([-float('inf'), 0])
    >>> optim.opt.set_min_objective(myfunc)
    >>> # [...]
    >>> optim.opt.set_xtol_rel(1e-4)
    >>> # don't need to pass the starting value as it is stored internally
    >>> optim.optimize()
    >>> minf = optim.opt.last_optimum_value()

    Note: if you train with bounds/nonlinear transforms, you may want
          to specify "null" forward and backward transformations for your
          parameters (i.e., the identity transformation), or otherwise those
          transformations may mix improperly.

    Attributes:
        opt: NLopt optimizer
        x0: initial value of trainable parameters
        loss_fn: original loss function
        loss: loss function for unbound 1D parameters
        grad_loss: gradients of loss function for unbound 1D parameters
        nlopt_loss: loss function compatible with NLopt
        ravel_backward: function to flatten and apply backward transformation to params
        unravel_forward: function to unflatten and apply forward to 1D trainable params
    """

    def __init__(
        self, state: ModelState, x: ArrayLike, y: ArrayLike, loss_fn: Callable, opt: int
    ) -> None:
        """
        Args:
            state: model state
            x: input data
            y: target data
            loss_fn: loss function
            opt: NLopt optimizer (e.g., nlopt.LD_MMA)
        """
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
        """optimize parameters

        Optimize the parameters using `self.x0` as starting
        point, and return the model state with the optimized
        values.
        """
        xt = self.opt.optimize(self.x0)
        params = self.unravel_forward(xt)
        state = self._state.update(dict(params=params))
        return state
