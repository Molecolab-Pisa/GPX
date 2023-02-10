import jax
import dataclasses
from typing import Callable


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Parameter:
    value: float
    trainable: bool
    transform_fn: Callable

    def tree_flatten(self):
        children = (self.value,)
        aux_data = (self.trainable, self.transform_fn)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
