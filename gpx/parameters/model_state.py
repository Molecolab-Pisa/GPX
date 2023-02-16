from .utils import _recursive_traverse_dict

import jax
import jax.numpy as jnp


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

    def _params_forward_transforms(self, params):
        return [p.forward_transform for p in _recursive_traverse_dict(params)]

    @property
    def params_forward_transforms(self):
        return self._params_forward_transforms(self.params)

    def _params_backward_transforms(self, params):
        return [p.backward_transform for p in _recursive_traverse_dict(params)]

    @property
    def params_backward_transforms(self):
        return self._params_backward_transforms(self.params)

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
            opt,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kernel, params_structure, opt = aux_data
        params = jax.tree_util.tree_unflatten(params_structure, children)
        return cls(kernel, params, **opt)

    def update(self, update_dict):
        kernel = (
            update_dict.pop("kernel") if "kernel" in update_dict.keys() else self.kernel
        )
        params = (
            update_dict.pop("params") if "params" in update_dict.keys() else self.params
        )
        opt = {entry: getattr(self, entry) for entry in self._register}
        for key, val in update_dict.items():
            opt[key] = val
        return self.__class__(kernel, params, **opt)
