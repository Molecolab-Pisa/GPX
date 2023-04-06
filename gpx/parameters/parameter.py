from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


@jax.tree_util.register_pytree_node_class
class Parameter:
    def __init__(
        self,
        value: ArrayLike,
        trainable: bool,
        forward_transform: Callable,
        backward_transform: Callable,
    ) -> None:
        self.value = jnp.array(value)
        self.trainable = trainable
        self.forward_transform = forward_transform
        self.backward_transform = backward_transform

    def __repr__(self) -> str:
        name = self.__class__.__name__
        reprstr = f"{name}(value={self.value}, trainable={self.trainable}"
        reprstr += f", forward_transform={self.forward_transform}"
        reprstr += f", backward_transform={self.backward_transform})"
        return reprstr

    def tree_flatten(self) -> Tuple[Array, Any]:
        children = (self.value,)
        aux_data = (self.trainable, self.forward_transform, self.backward_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: ArrayLike) -> "Parameter":
        return cls(*children, *aux_data)


def parse_param(param: Tuple[ArrayLike, bool, Callable, Callable]) -> Parameter:
    errmsg = "Provide each parameter as a 4-tuple"
    errmsg += " (value: float|jax.jnp.ndarray, trainable: bool,"
    errmsg += " forward: callable, backward: callable)"
    try:
        value, trainable, forward, backward = param
    except TypeError as e:
        raise TypeError(f"{e}. {errmsg}") from None

    if not (isinstance(value, (np.ndarray, Array)) or np.isscalar(value)):
        raise TypeError(f"Expected arraylike input, got {value}. {errmsg}")

    if not isinstance(trainable, bool):
        raise TypeError(f"Expected boolean input, got {trainable}. {errmsg}")

    if not callable(forward):
        raise TypeError(f"Expected callable input, got {forward}. {errmsg}")

    if not callable(backward):
        raise TypeError(f"Expected callable input, got {backward}. {errmsg}")

    return Parameter(
        value=value,
        trainable=trainable,
        forward_transform=forward,
        backward_transform=backward,
    )
