from __future__ import annotations

from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class Parameter:
    def __init__(
        self,
        value: Union[float, jnp.ndarray],
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

    def tree_flatten(self) -> Tuple[jnp.ndarray, Any]:
        children = (self.value,)
        aux_data = (self.trainable, self.forward_transform, self.backward_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: jnp.ndarray) -> "Parameter":
        return cls(*children, *aux_data)


def parse_param(param: Tuple[jnp.ndarray, bool, Callable, Callable]) -> Parameter:
    errmsg = "Provide each parameter as a 4-tuple"
    errmsg += " (value: float|jax.jnp.ndarray, trainable: bool,"
    errmsg += " forward: callable, backward: callable)"
    try:
        value, trainable, forward, backward = param
    except TypeError as e:
        raise TypeError(f"{e}. {errmsg}") from None

    if not isinstance(value, float) and not isinstance(value, jnp.ndarray):
        raise RuntimeError(f"You provided value as {type(value)}. {errmsg}")

    if not isinstance(trainable, bool):
        raise RuntimeError(f"You provided trainable as {type(trainable)}. {errmsg}")

    if not callable(forward):
        raise RuntimeError(f"You provided forward as {type(forward)}. {errmsg}")

    if not callable(backward):
        raise RuntimeError(f"You provided backward as {type(backward)}. {errmsg}")

    return Parameter(
        value=value,
        trainable=trainable,
        forward_transform=forward,
        backward_transform=backward,
    )
