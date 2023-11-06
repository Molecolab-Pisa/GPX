from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array, random
from jax._src import prng
from jax.typing import ArrayLike

from ..optimizers import scipy_minimize, scipy_minimize_derivs
from ..parameters import ModelState


def _check_object_is_callable(obj: Any, name: str) -> None:
    if not callable(obj):
        raise ValueError(f"{name} must be a callable, you provided {type(obj)}")


def _check_object_is_type(obj: Any, ref_type: Any, name: str) -> None:
    if not isinstance(obj, ref_type):
        raise ValueError(
            f"{name} must be a {ref_type} instance, you provided {type(obj)}"
        )


def _check_recursive_dict_type(dictionary: Dict, ref_type: Any) -> None:
    for key, value in dictionary.items():
        if isinstance(value, dict):
            _check_recursive_dict_type(value, ref_type=ref_type)
        else:
            if not isinstance(value, ref_type):
                raise ValueError(
                    f"{key} must be a {ref_type} instance, you provided {type(value)}"
                )


def sample(
    key: prng.PRNGKeyArray,
    mean: ArrayLike,
    cov: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    if mean.ndim > 1:
        samples = []
        for dim in range(mean.shape[1]):
            subkey, key = random.split(key)
            sample = random.multivariate_normal(
                key=key,
                mean=mean[:, dim],
                cov=cov,
                shape=(n_samples,),
            )
            samples.append(sample)
    else:
        sample = random.multivariate_normal(key=key, mean=mean, cov=cov)
    return jnp.array(sample)


# ============================================================================
# Loss function minimization with randomized restarts
# ============================================================================


def _check_random_key(key: prng.PRNGKeyArray, num_restarts: int):
    if num_restarts > 0 and key is None:
        raise ValueError(
            "If you want to train with randomized restarts, please provide"
            f"a valid JAX PRNGKey ({key} is not valid)."
        )


def randomized_minimization(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    minimization_function: Callable = scipy_minimize,
    num_restarts: Optional[int] = 0,
    return_history: Optional[bool] = False,
    opt_kwargs: Dict[str, Any] = None,
) -> ModelState:
    """performes one minimization of the loss function and then it applies
    the randomization function before performing the subsequent optimization.
    This is repeated for a number of times in range(num_restarts).

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        y: target
        minimization_function: loss minimization algorithm (from optimizers).
                               Default is 'scipy_minimize'. In RBFNet model it
                               can also be 'optax_minimize'
        num_restarts: number of restarts after the first optimization

    Returns:
        state: model state
        *optres: tuple with optimization results and other possibile outputs
                 (for example, optax optimizer returns the history of the loss function)
        states: list of model states at each restart.
        losses: list of loss values at each restart.
    """
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs

    _check_random_key(key=key, num_restarts=num_restarts)

    states = []
    losses = []
    opt_info = []

    state, *optres = minimization_function(
        state=state, x=x, y=y, loss_fn=state.loss_fn, **opt_kwargs
    )
    loss = state.loss_fn(state=state, x=x, y=y)

    states.append(state)
    losses.append(loss)
    opt_info.append(optres)

    for _restart in range(num_restarts):
        subkey, key = jax.random.split(key)
        state = state.randomize(key)

        state, *optres = minimization_function(
            state=state, x=x, y=y, loss_fn=state.loss_fn, **opt_kwargs
        )
        loss = state.loss_fn(state=state, x=x, y=y)

        states.append(state)
        losses.append(loss)
        opt_info.append(optres)

    idx = losses.index(min(losses))
    state = states[idx]
    optres = opt_info[idx]

    if return_history:
        return state, *optres, states, losses
    else:
        return state, *optres


def randomized_minimization_derivs(
    key: prng.PRNGKeyArray,
    state: ModelState,
    x: ArrayLike,
    y: ArrayLike,
    jacobian: ArrayLike,
    minimization_function: Callable = scipy_minimize_derivs,
    num_restarts: Optional[int] = 0,
    return_history: Optional[bool] = False,
    opt_kwargs: Dict[str, Any] = None,
) -> ModelState:
    """performes one minimization of the loss function and then it applies
    the randomization function before performing the subsequent optimization.
    This is repeated for a number of times in range(num_restarts).

    Args:
        key: JAX PRNGKey
        state: model state
        x: observations
        y: target
        jacobian: jacobian of x
        minimization_function: loss minimization algorithm (from optimizers).
                               Default is 'scipy_minimize'. In RBFNet model it
                               can also be 'optax_minimize'
        num_restarts: number of restarts after the first optimization

    Returns:
        state: model state
        *optres: tuple with optimization results and other possibile outputs
                 (for example, optax optimizer returns the history of the loss function)
        states: list of model states at each restart.
        losses: list of loss values at each restart.
    """
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs

    _check_random_key(key=key, num_restarts=num_restarts)

    states = []
    losses = []
    opt_info = []

    state, *optres = minimization_function(
        state=state, x=x, y=y, jacobian=jacobian, loss_fn=state.loss_fn, **opt_kwargs
    )
    loss = state.loss_fn(state=state, x=x, y=y, jacobian=jacobian)

    states.append(state)
    losses.append(loss)
    opt_info.append(optres)

    for _restart in range(num_restarts):
        subkey, key = jax.random.split(key)
        state = state.randomize(key)

        state, *optres = minimization_function(
            state=state,
            x=x,
            y=y,
            jacobian=jacobian,
            loss_fn=state.loss_fn,
            **opt_kwargs,
        )
        loss = state.loss_fn(
            state=state,
            x=x,
            y=y,
            jacobian=jacobian,
        )

        states.append(state)
        losses.append(loss)
        opt_info.append(optres)

    idx = losses.index(min(losses))
    state = states[idx]
    optres = opt_info[idx]

    if return_history:
        return state, *optres, states, losses
    else:
        return state, *optres
