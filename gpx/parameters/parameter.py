import dataclasses
from typing import Callable


# currently not registrable as a PyTree, as
# we want to treat a Parameter as a leaf when
# flattening a dictioanry of Parameter instances
# in order to get its auxiliary data

# @jax.tree_util.register_pytree_node_class
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
