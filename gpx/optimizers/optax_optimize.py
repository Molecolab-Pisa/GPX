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
from collections import defaultdict
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Tuple

import jax
from jax import Array
from jax.typing import ArrayLike
from tqdm import tqdm

from ..parameters import ModelState

try:
    import optax
except ImportError:
    warnings.warn(
        "optax is not installed. Interface to optax optimizers is not available.",
        stacklevel=2,
    )

# state of the optax optimizer
OptimizerState = Any

# optax optimizer
GradientTransformation = Any


def train_with_constrained_parameters(loop_func: Callable) -> Callable:
    """Make an optax train loop work with bounded parameters

    This decorator transforms an optax train loop so that it
    is compatible with the constraints of the GPX parameters.
    This means that one can write a simple optax training loop
    without worrying about the fact that parameters are actually
    bounded (by their forward/backward transformations).
    The version of train loop compatible with the parameter transformation
    is then obtained as:

        bounded_loop_func = train_with_constrained_parameters(loop_func)

    The requirement to use this decorator is that the loop function
    should have the following signature:

        loop_func(state, loss_fn, *args, **kwargs) -> Tuple[state, ...]

    i.e., the ModelState and the loss functions are the first and
    second arguments, and the first return value should be the optimized
    ModelState.

    This decorator makes the optimization work in the unbound space
    of the parameters values, while constraining parameters when calling
    the loss and when returning the optimized state.
    """

    @wraps(loop_func)
    def wrapper(state, loss_fn, *args, **kwargs):
        # train in the unbound space
        state = state.transform(mode="backward")

        @jax.jit
        def _loss_fn(state, x, y):
            # bound parameters before calling the loss
            state = state.transform(mode="forward")
            return loss_fn(state, x, y)

        state, *res = loop_func(state, _loss_fn, *args, **kwargs)

        # state in bounded space
        state = state.transform(mode="forward")

        return state, *res

    return wrapper


@partial(jax.jit, static_argnums=(2, 3))
def _step(
    opt_state: OptimizerState,
    state: ModelState,
    optimizer: GradientTransformation,
    loss_fn: Callable[ModelState, ArrayLike, ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
) -> Tuple[OptimizerState, ModelState, Array]:
    grad_loss = jax.value_and_grad(loss_fn)

    loss, grads = grad_loss(state, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    state = optax.apply_updates(state, updates)

    return opt_state, state, loss


@train_with_constrained_parameters
def _optimize(
    state: ModelState,
    loss_fn: Callable[ModelState, ArrayLike, ArrayLike],
    optimizer: GradientTransformation,
    x: ArrayLike,
    y: ArrayLike,
    nsteps: int,
    update_every: int,
    **kwargs,
) -> Tuple[ModelState, OptimizerState, Dict]:
    opt_state = optimizer.init(state)
    history = defaultdict(list)
    iterator = tqdm(range(nsteps), ncols=100)
    for i in iterator:
        opt_state, state, loss = _step(opt_state, state, optimizer, loss_fn, x, y)
        history["train_loss"].append(loss)
        if i % update_every == 0:
            iterator.set_postfix(loss=f"{loss:.6f}")
    return state, opt_state, history


@partial(jax.jit, static_argnums=(2, 3))
def _step_with_validation(
    opt_state: OptimizerState,
    state: ModelState,
    optimizer: GradientTransformation,
    loss_fn: Callable[ModelState, ArrayLike, ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
) -> Tuple[OptimizerState, ModelState, Array, Array]:
    grad_loss = jax.value_and_grad(loss_fn)

    loss, grads = grad_loss(state, x, y)
    loss_val = loss_fn(state, x_val, y_val)
    updates, opt_state = optimizer.update(grads, opt_state)
    state = optax.apply_updates(state, updates)

    return opt_state, state, loss, loss_val


@train_with_constrained_parameters
def _optimize_with_validation(
    state: ModelState,
    loss_fn: Callable,
    optimizer: GradientTransformation,
    x: ArrayLike,
    y: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    nsteps: int,
    update_every: int,
    **kwargs,
) -> Tuple[ModelState, OptimizerState, Dict]:
    opt_state = optimizer.init(state)
    history = defaultdict(list)
    iterator = tqdm(range(nsteps), ncols=100)
    for i in iterator:
        opt_state, state, loss, loss_val = _step_with_validation(
            opt_state, state, optimizer, loss_fn, x, y, x_val, y_val
        )
        history["train_loss"].append(loss)
        history["valid_loss"].append(loss_val)
        if i % update_every == 0:
            iterator.set_postfix(loss=f"{loss:.6f}", loss_val=f"{loss_val:.6f}")
    return state, opt_state, history


def _check_optimizer(
    optimizer: GradientTransformation, learning_rate: float
) -> GradientTransformation:
    if optimizer is None:
        optimizer = optax.adam(learning_rate=learning_rate)
    return optimizer


def _check_validation(x_val: ArrayLike, y_val: ArrayLike) -> Callable:
    val_given = x_val is not None or y_val is not None
    both_val_given = x_val is not None and y_val is not None

    if val_given and not both_val_given:
        raise ValueError(
            "If you want to compute the validation loss, please provide"
            " both x_val and y_val"
        )
    elif both_val_given:
        optim_fn = _optimize_with_validation
    else:
        optim_fn = _optimize
    return optim_fn


def optax_minimize(
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    loss_fn: Callable[ModelState, ArrayLike, ArrayLike],
    optimizer: GradientTransformation = None,
    x_val: ArrayLike = None,
    y_val: ArrayLike = None,
    nsteps: int = 10,
    learning_rate: float = 1.0,
    update_every: int = 1,
) -> Tuple[ModelState, OptimizerState, Dict[str, List]]:
    """Minimize the loss with optax optimizers

    Args:
        state: ModelState
        x: train input
        y: train target
        loss_fn: loss function
        optimizer: optax optimizer, default is adam
        x_val: validation input
        y_val: validation target
        nsteps: number of train steps (epochs)
        learning_rate: optimizer learning rate, used only if optimizer is None
        update_every: update frequency of the progress bar
    Returns:
        state: optimized ModelState
        opt_state: state of the optax optimizer
        history: loss history
    """
    optimizer = _check_optimizer(optimizer, learning_rate=learning_rate)
    optim_fn = _check_validation(x_val=x_val, y_val=y_val)

    state, opt_state, history = optim_fn(
        state=state,
        loss_fn=loss_fn,
        optimizer=optimizer,
        x=x,
        y=y,
        nsteps=nsteps,
        update_every=update_every,
        x_val=x_val,
        y_val=y_val,
    )

    return state, opt_state, history
