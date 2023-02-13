import jax
import dataclasses
from typing import Callable


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

    def tree_flatten(self):
        children = (self.value,)
        aux_data = (self.trainable, self.forward_transform, self.backward_transform)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
