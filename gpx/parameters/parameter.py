from typing import Any, Tuple, Callable

import jax
import dataclasses

Array = Any


# currently not registrable as a PyTree, as
# we want to treat a Parameter as a leaf when
# flattening a dictioanry of Parameter instances
# in order to get its auxiliary data


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Parameter:
    value: float
    trainable: bool
    forward_transform: Callable
    backward_transform: Callable

    def tree_flatten(self) -> Tuple[Array, Any]:
        children = (self.value,)
        aux_data = (self.trainable, self.forward_transform, self.backward_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Array) -> "Parameter":
        return cls(*children, *aux_data)


def parse_param(param: Tuple[Array, bool, Callable, Callable]) -> Parameter:
    errmsg = "Provide each parameter as a 4-tuple (value: float|jax.Array, trainable: bool, forward: callable, backward: callable)"
    try:
        value, trainable, forward, backward = param
    except TypeError as e:
        raise TypeError(f"{e}. {errmsg}") from None

    if not isinstance(value, float) and not isinstance(value, jax.Array):
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
