from .parameter import Parameter
from .utils import _recursive_traverse_dict

import jax
import jax.numpy as jnp

# from typing import Callable, Dict


@jax.tree_util.register_pytree_node_class
class ModelState:
    def __init__(self, kernel, params, **kwargs):
        self.kernel = kernel
        self.params = params

        self._register = []
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._register_entry(name)

    def _register_entry(self, entry):
        self._register.append(entry)

    def __repr__(self):
        rep = f"{self.__class__.__name__}(kernel={self.kernel}, params={self.params})"
        if len(self._register) != 0:
            rep = rep[:-1]
            for entry in self._register:
                rep += f", {entry}={getattr(self, entry)}"
            rep += ")"
        return rep

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
        p_structure = jax.tree_util.tree_structure(p)
        self._params = jax.tree_util.tree_unflatten(p_structure, values)
        self._params_structure = p_structure

    def _params_transform_fns(self, params):
        return [p.transform_fn for p in _recursive_traverse_dict(params)]

    @property
    def params_transform_fns(self):
        return self._params_transform_fns(self.params)

    def _params_value(self, params):
        return jnp.array([p.value for p in _recursive_traverse_dict(params)])

    @property
    def params_value(self):
        return self._params_value(self.params)

    def _params_trainable(self, params):
        return jnp.array([p.trainable for p in _recursive_traverse_dict(params)])

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

        opt = {}
        if len(self._register) != 0:
            for entry in self._register:
                opt[entry] = getattr(self, entry)

        return (params_value), (
            self.kernel,
            self._params_structure,
            params_trainable,
            params_transforms,
            opt,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kernel, params_structure, params_trainable, params_transforms, opt = aux_data
        params = tuple(
            Parameter(v, t, f)
            for v, t, f in zip(children, params_trainable, params_transforms)
        )
        params = jax.tree_util.tree_unflatten(params_structure, children)
        return cls(kernel, params, **opt)

    def update_state(self, entry, value):
        # currently only works for optional arguments
        kernel = self.kernel
        params = self.params
        opt = {entry: getattr(self, entry) for entry in self._register}
        opt[entry] = value
        return self.__class__(kernel, params, **opt)
