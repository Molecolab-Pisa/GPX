from .parameter import Parameter

import jax
import jax.numpy as jnp
import dataclasses
from typing import Callable, Dict


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class ModelState:
    kernel: Callable
    params: Dict

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, k):
        self._kernel = k

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        trainable = self._params_trainable(p)
        values = self._params_value(p)
        values = jnp.where(trainable, values, jax.lax.stop_gradient(values))
        transform_fns = self._params_transform_fns(p)
        params = tuple(
            Parameter(v, t, f) for v, t, f in zip(values, trainable, transform_fns)
        )
        p_structure = jax.tree_util.tree_structure(p)
        self._params = jax.tree_util.tree_unflatten(p_structure, params)
        self._params_structure = p_structure

    def _params_transform_fns(self, params):
        return [p.transform_fn for p in jax.tree_util.tree_leaves(params)]

    @property
    def params_transform_fns(self):
        return self._params_transform_fns(self.params)

    def _params_value(self, params):
        return jnp.array([p.value for p in jax.tree_util.tree_leaves(params)])

    @property
    def params_value(self):
        return self._params_value(self.params)

    def _params_trainable(self, params):
        return jnp.array([p.trainable for p in jax.tree_util.tree_leaves(params)])

    @property
    def params_trainable(self):
        return self._params_trainable(self.params)

    def tree_flatten(self):
        params_value = self.params_value
        params_trainable = self.params_trainable
        params_transforms = self.params_transform_fns
        params_value = jnp.where(
            params_trainable, params_value, jax.lax.stop_gradient(params_value)
        )
        return (params_value), (
            self.kernel,
            self._params_structure,
            params_trainable,
            params_transforms,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kernel, params_structure, params_trainable, params_transforms = aux_data
        params = tuple(
            Parameter(v, t, f)
            for v, t, f in zip(children, params_trainable, params_transforms)
        )
        params = jax.tree_util.tree_unflatten(params_structure, params)
        return cls(kernel, params)
