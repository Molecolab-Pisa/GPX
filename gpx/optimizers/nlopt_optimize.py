from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from jax import grad, jit
from jax.typing import ArrayLike

from ..parameters import ModelState
from .utils import ravel_backward_trainables, unravel_forward_trainables

Self = Any

try:
    import nlopt
except ImportError:
    warnings.warn(
        "NLopt is not installed. Interface to NLopt optimizers is not available.",
        stacklevel=2,
    )


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

    >>> optim = NLoptWrapper(state, opt=nlopt.LD_MMA)
    >>> # you access the NLopt optimizer as `optim.opt`
    >>> optim.opt.set_lower_bounds([-float('inf'), 0])
    >>> # [...]
    >>> optim.opt.set_xtol_rel(1e-4)
    >>> # the set_min_objective is set internally
    >>> optim_state = optim.optimize(state, x, y, loss_fn)
    >>> minf = optim.opt.last_optimum_value()

    Note: if you train with bounds/nonlinear transforms, you may want
          to specify "null" forward and backward transformations for your
          parameters (i.e., the identity transformation), or otherwise those
          transformations may mix improperly.

    Attributes:
        opt: NLopt optimizer
        orig_x0: initial value of trainable parameters
        orig_state: ModelState used to instantiate the optimizer
        ravel_backward: function to flatten and apply backward transformation to params
        unravel_forward: function to unflatten and apply forward to 1D trainable params
    """

    def __init__(self, state: ModelState, opt: int) -> None:
        """
        Args:
            state: model state
            opt: NLopt optimizer (e.g., nlopt.LD_MMA)
        """
        self.orig_state = state

        # functions to unbound/bound parameters
        x0, tdef, unravel_fn = ravel_backward_trainables(state.params)
        unravel_forward = unravel_forward_trainables(unravel_fn, tdef, state.params)
        # unbound parameters, flattened to 1D
        self.orig_x0 = x0
        # function to unbound and flatten
        self.ravel_backward = ravel_backward_trainables
        # function to bound + reconstruct
        self.unravel_forward = unravel_forward

        # NLopt optimizer
        self.opt = nlopt.opt(opt, len(x0))

    def optimize(
        self, state: ModelState, x: ArrayLike, y: ArrayLike, loss_fn: Callable
    ) -> ModelState:
        """optimize parameters

        Args:
            state: ModelState, starting value of parameters are taken from here.
                   Important: Must be the same (except for parameter values) as
                              the state used to initialize the wrapper.
            x: input data
            y: target data
            loss_fn: loss function, signature fn(state, x, y)
        Returns:
            optim_state: optimized state
            optim_result: NLopt optimization return code
        """
        # set the starting point from state
        x0, _, _ = self.ravel_backward(state.params)

        # setup the loss function
        @jit
        def loss(xt):
            # go in bound space and reconstruct params
            params = self.unravel_forward(xt)
            ustate = state.update(dict(params=params))
            return loss_fn(ustate, x, y)

        grad_loss = jit(grad(loss))

        # wrap the loss so it's compatible with NLopt
        def nlopt_loss(xt, grad):
            if grad.size > 0:
                # gradients from JAX
                _grad = grad_loss(xt)
                # in-place update
                grad[:] = np.array(_grad)
            return np.array(loss(xt)) * 1.0  # the 1.0 is a hack to a silent issue

        # set loss to minimize
        self.opt.set_min_objective(nlopt_loss)

        # minimize
        xt = self.opt.optimize(x0)

        # reconstruct parameters and state
        params = self.unravel_forward(xt)
        state = state.update(dict(params=params))
        return state, self.opt.last_optimize_result()
