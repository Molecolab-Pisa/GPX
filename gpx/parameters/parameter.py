from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax._src import prng
from jax.typing import ArrayLike

from .utils import _check_same_dtype, _check_same_shape


@jax.tree_util.register_pytree_node_class
class Parameter:
    def __init__(
        self,
        value: ArrayLike,
        trainable: bool,
        forward_transform: Callable,
        backward_transform: Callable,
        prior: Callable,
    ) -> None:
        self.value = jnp.array(value)
        self.trainable = trainable
        self.forward_transform = forward_transform
        self.backward_transform = backward_transform
        self.prior = prior

        # check that dtype and shape of value and prior match
        _check_same_shape(self.value, self.prior, "value", "prior")
        _check_same_dtype(self.value, self.prior, "value", "prior")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        reprstr = f"{name}(value={self.value}, trainable={self.trainable}"
        reprstr += f", forward_transform={self.forward_transform}"
        reprstr += f", backward_transform={self.backward_transform}"
        reprstr += f", prior={self.prior})"
        return reprstr

    def tree_flatten(self) -> Tuple[Array, Any]:
        children = (self.value,)
        aux_data = (
            self.trainable,
            self.forward_transform,
            self.backward_transform,
            self.prior,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: ArrayLike) -> "Parameter":
        return cls(*children, *aux_data)

    def update(self, update_dict: Dict) -> "Parameter":
        value = (
            update_dict.pop("value") if "value" in update_dict.keys() else self.value
        )
        trainable = (
            update_dict.pop("trainable")
            if "trainable" in update_dict.keys()
            else self.trainable
        )
        forward_transform = (
            update_dict.pop("forward_transform")
            if "forward_transform" in update_dict.keys()
            else self.forward_transform
        )
        backward_transform = (
            update_dict.pop("backward_transform")
            if "backward_transform" in update_dict.keys()
            else self.backward_transform
        )
        prior = (
            update_dict.pop("prior") if "prior" in update_dict.keys() else self.prior
        )
        return self.__class__(
            value=value,
            trainable=trainable,
            forward_transform=forward_transform,
            backward_transform=backward_transform,
            prior=prior,
        )

    def __copy__(self):
        # calling update with an empty dictionary is equivalent
        # to creating a new instance of the class with the same
        # attributes
        return self.update({})

    def copy(self) -> "Parameter":
        "returns a shallow copy of the parameter"
        return copy.copy(self)

    def sample_prior(self, key: prng.PRNGKeyArray) -> "Parameter":
        "returns a new parameter with value sampled from the prior"
        sample = self.prior.sample(key)
        # constrain the sampled value to the enforced range
        sample = self.forward_transform(sample)
        update_dict = dict(value=sample)
        return self.update(update_dict)


def is_parameter(p: Any) -> bool:
    "True if p is a Parameter instance, False otherwise"
    return isinstance(p, Parameter)
