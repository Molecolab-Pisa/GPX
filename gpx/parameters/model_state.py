from __future__ import annotations
from typing import Dict, Callable, List, Optional, Tuple, Any

from .utils import _recursive_traverse_dict
from .parameter import Parameter

import jax
import jax.numpy as jnp

import numpy as np
from tabulate import tabulate


@jax.tree_util.register_pytree_node_class
class ModelState:
    def __init__(
        self, kernel: Callable, params: Dict[str, Parameter], **kwargs: Any
    ) -> None:
        self.kernel = kernel
        self.params = params

        self._register = []
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._register_entry(name)

    def _register_entry(self, entry: str) -> None:
        self._register.append(entry)

    def __repr__(self) -> str:
        rep = f"{self.__class__.__name__}(kernel={self.kernel}, params={self.params})"
        if len(self._register) != 0:
            rep = rep[:-1]
            for entry in self._register:
                rep += f", {entry}={getattr(self, entry)}"
            rep += ")"
        return rep

    @property
    def kernel(self) -> Callable:
        return self._kernel

    @kernel.setter
    def kernel(self, k: Callable) -> None:
        self._kernel = k

    @property
    def params(self) -> Dict[str, Parameter]:
        return self._params

    @params.setter
    def params(self, p: Dict[str, Parameter]) -> None:
        trainable = self._params_trainable(p)
        values = self._params_value(p)
        # values = jnp.where(trainable, values, jax.lax.stop_gradient(values))
        values = [
            v if t else jax.lax.stop_gradient(v) for t, v in zip(trainable, values)
        ]
        p_structure = jax.tree_util.tree_structure(p)
        self._params = jax.tree_util.tree_unflatten(p_structure, values)
        self._params_structure = p_structure

    def _params_forward_transforms(
        self, params: Dict[str, Parameter]
    ) -> List[Callable]:
        return [p.forward_transform for p in _recursive_traverse_dict(params)]

    @property
    def params_forward_transforms(self) -> List[Callable]:
        return self._params_forward_transforms(self.params)

    def _params_backward_transforms(
        self, params: Dict[str, Parameter]
    ) -> List[Callable]:
        return [p.backward_transform for p in _recursive_traverse_dict(params)]

    @property
    def params_backward_transforms(self) -> List[Callable]:
        return self._params_backward_transforms(self.params)

    def _params_value(self, params: Dict[str, Parameter]) -> List[jnp.ndarray]:
        return [p.value for p in _recursive_traverse_dict(params)]

    @property
    def params_value(self) -> List[jnp.ndarray]:
        return self._params_value(self.params)

    def _params_trainable(self, params: Dict[str, Parameter]) -> List[jnp.ndarray]:
        return [p.trainable for p in _recursive_traverse_dict(params)]

    @property
    def params_trainable(self) -> List[jnp.ndarray]:
        return self._params_trainable(self.params)

    def tree_flatten(self) -> Tuple[jnp.ndarray, Any]:
        params_value = self.params_value
        params_trainable = self.params_trainable
        params_value = [
            v if t else jax.lax.stop_gradient(v)
            for t, v in zip(params_trainable, params_value)
        ]
        # params_value = jnp.where(
        #    params_trainable, params_value, jax.lax.stop_gradient(params_value)
        # )

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
    def tree_unflatten(cls, aux_data: Any, children: jnp.ndarray) -> "ModelState":
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

    def print_params(self, tablefmt: Optional[str] = "simple_grid") -> None:
        params = self.params.copy()
        kernel_params = params.pop("kernel_params")

        headers = ["name", "type", "dtype", "shape", "trainable", "value"]

        def string_repr(p):
            return np.array2string(p.value, edgeitems=1, threshold=1)

        def get_info(p):
            v = p.value
            return (type(v), v.dtype, v.shape, p.trainable, string_repr(p))

        fields = [["kernel " + k] + list(get_info(p)) for k, p in kernel_params.items()]
        fields += [[k] + list(get_info(p)) for k, p in params.items()]

        with np.printoptions(edgeitems=0):
            print(tabulate(fields, headers=headers, tablefmt=tablefmt))
