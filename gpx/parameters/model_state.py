from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src import prng
from jax.tree_util import (
    tree_flatten,
    tree_leaves,
    tree_leaves_with_path,
    tree_unflatten,
)
from jax.typing import ArrayLike
from tabulate import tabulate

from .parameter import Parameter, is_parameter
from .utils import _flatten_dict, _is_numeric, _unflatten_dict


@jax.tree_util.register_pytree_node_class
class ModelState:
    def __init__(
        self,
        kernel: Callable,
        mean_function: Callable,
        params: Dict[str, Parameter],
        **kwargs: Any,
    ) -> None:
        self.kernel = kernel
        self.mean_function = mean_function
        self.params = params

        self._register = []
        for name, value in kwargs.items():
            if _is_numeric(value):
                value = jnp.asarray(value)
            setattr(self, name, value)
            self._register_entry(name)

    def _register_entry(self, entry: str) -> None:
        self._register.append(entry)

    def __repr__(self) -> str:
        rep = f"{self.__class__.__name__}(kernel={self.kernel}, mean_function={self.mean_function}, params={self.params})"  # noqa: E501
        if len(self._register) != 0:
            rep = rep[:-1]
            for entry in self._register:
                rep += f", {entry}={getattr(self, entry)}"
            rep += ")"
        return rep

    @property
    def kernel(self) -> Callable:
        return self._kernel

    # TODO Edo: is this really important?
    @kernel.setter
    def kernel(self, k: Callable) -> None:
        self._kernel = k

    @property
    def mean_function(self) -> Callable:
        return self._mean_function

    @mean_function.setter
    def mean_function(self, m: Callable) -> None:
        self._mean_function = m

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

    def _params_priors(
        self, params: Dict[str, Parameter]
    ) -> List["Prior"]:  # noqa: F821
        return [p.prior for p in tree_leaves(params, is_leaf=is_parameter)]

    @property
    def params_priors(self) -> List["Prior"]:  # noqa: F821
        return self._params_priors(self.params)

    def _params_forward_transforms(
        self, params: Dict[str, Parameter]
    ) -> List[Callable]:
        return [p.bijector.forward for p in tree_leaves(params, is_leaf=is_parameter)]

    @property
    def params_forward_transforms(self) -> List[Callable]:
        return self._params_forward_transforms(self.params)

    def _params_backward_transforms(
        self, params: Dict[str, Parameter]
    ) -> List[Callable]:
        return [p.bijector.backward for p in tree_leaves(params, is_leaf=is_parameter)]

    @property
    def params_backward_transforms(self) -> List[Callable]:
        return self._params_backward_transforms(self.params)

    def _params_value(self, params: Dict[str, Parameter]) -> List[Array]:
        return [p.value for p in tree_leaves(params, is_leaf=is_parameter)]

    @property
    def params_value(self) -> List[Array]:
        return self._params_value(self.params)

    def _params_trainable(self, params: Dict[str, Parameter]) -> List[Array]:
        return [p.trainable for p in tree_leaves(params, is_leaf=is_parameter)]

    @property
    def params_trainable(self) -> List[Array]:
        return self._params_trainable(self.params)

    def tree_flatten(self) -> Tuple[Array, Any]:
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
            self.mean_function,
            self._params_structure,
            opt,
        )

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: ArrayLike) -> "ModelState":
        kernel, mean_function, params_structure, opt = aux_data
        params = jax.tree_util.tree_unflatten(params_structure, children)
        return cls(kernel, mean_function, params, **opt)

    def update(self, update_dict):
        kernel = (
            update_dict.pop("kernel") if "kernel" in update_dict.keys() else self.kernel
        )
        mean_function = (
            update_dict.pop("mean_function")
            if "mean_function" in update_dict.keys()
            else self.mean_function
        )
        params = (
            update_dict.pop("params") if "params" in update_dict.keys() else self.params
        )
        opt = {entry: getattr(self, entry) for entry in self._register}
        for key, val in update_dict.items():
            opt[key] = val
        return self.__class__(kernel, mean_function, params, **opt)

    def __copy__(self) -> "ModelState":
        return self.update({})

    def copy(self) -> "ModelState":
        return copy.copy(self)

    def print_params(self, tablefmt: Optional[str] = "simple_grid") -> None:
        params = self.params.copy()
        kernel_params = params.pop("kernel_params")

        headers = [
            "name",
            "trainable",
            "bijector",
            "prior",
            # "type",
            "dtype",
            "shape",
            "value",
        ]

        # colored headers
        headers = ["\033[1;35m" + h + "\033[0m" for h in headers]

        def string_repr(p):
            return np.array2string(p.value, edgeitems=1, threshold=1)

        def get_info(p):
            v = p.value
            return (
                p.trainable,
                str(p.bijector),
                str(p.prior),
                # type(v).__name__,
                v.dtype,
                v.shape,
                string_repr(p),
            )

        fields = []
        for k, p in tree_leaves_with_path(kernel_params, is_leaf=is_parameter):
            name = "kernel " + ":".join([sk.key for sk in k])
            # name = "kernel " + k[-1].key
            fields.append([name] + list(get_info(p)))

        fields += [[k] + list(get_info(p)) for k, p in params.items()]

        with np.printoptions(edgeitems=0):
            print(tabulate(fields, headers=headers, tablefmt=tablefmt))

    def save(self, state_file: str) -> Dict:
        """Saves the state values to file

        Saves the state values to a file "state_file".
        Note that this functions only saves the values of
        the Parameter instances that are stored in the model state.

        Auxiliary data stored in the model state is saved as a
        numpy array if possible. Otherwise, an exception is raised.

        Args:
            state_file: path to the output state file.
        Returns:
            saved_dict: dictionary of values saved to state_file.
        """
        # Get the parameters
        params = self.params.copy()

        # Distinguish between kernel parameters and other parameters
        # everything here must be an instance of the Parameter class
        # Prepend "params:" or "params:kernel_params:" to identify
        # these as parameters in the saved dictionary
        kernel_params = params.pop("kernel_params")
        kernel_params = _flatten_dict(
            kernel_params,
            starting_key="params:kernel_params",
            sep=":",
            map_value=lambda p: p.value,
        )
        params = _flatten_dict(
            params, starting_key="params", sep=":", map_value=lambda p: p.value
        )

        # Join
        params |= kernel_params

        # Get the auxiliary attributes
        for opt in self._register:
            aux_opt = getattr(self, opt)

            # do not dump callables
            if callable(aux_opt):
                continue

            params[opt] = aux_opt

        # Dump
        np.savez(state_file, **params)

        return params

    def load(self, state_file: str) -> "ModelState":  # noqa: C901
        """Loads the state values from file

        Loads the state values stored in "state_file". A new
        instance of ModelState is returned.

        Args:
            state_file: path to the input state file.
        Returns:
            new_state: new ModelState instance with values loaded
                       from state_file.
        """
        # Load the new state values
        dumped = np.load(state_file, allow_pickle=True)
        dumped = {name: value for name, value in dumped.items()}

        dumped_dict = _unflatten_dict(dumped, sep=":")

        # =============================================
        # Parameters are handled here
        dumped_params = dumped_dict.pop("params")

        # Now they should have the same structure, so this operation
        # is consistent
        new_values = tree_leaves(dumped_params)
        params, params_def = tree_flatten(
            self.params, is_leaf=lambda p: isinstance(p, Parameter)
        )

        if len(new_values) != len(params):
            raise ValueError("Wrong number of parameters in dumped file.")

        params = [p.update({"value": v}) for p, v in zip(params, new_values)]
        params = tree_unflatten(params_def, params)
        # =============================================

        # =============================================
        # Check on optional/auxiliary data happen here
        for name, value in dumped_dict.items():
            # identify booleans
            if np.issubdtype(value.dtype, np.bool_) and value.ndim == 0:
                dumped_dict[name] = value.item()

            # identify None types
            elif value.ndim == 0:
                dumped_dict[name] = None if value.item() is None else value
        # =============================================

        update_dict = {"params": params} | dumped_dict

        return self.update(update_dict)

    def randomize(
        self, key: prng.PRNGKeyArray, opt: Dict[str, Any] = None
    ) -> "ModelState":
        """Creates a new state with randomized parameter values

        Creates a copy of the current state and updates the copy so that the parameter
        values are sampled from their prior distribution and the other values are set
        to default.
        """

        # Init dictionary for optional parameters
        opt = {} if opt is None else opt

        # Make a copy of the current model state
        state = self.copy()

        trainables = state.params_trainable
        priors = state.params_priors
        forwards = state.params_forward_transforms

        # Flatten the parameters dictionary so that
        # all the keys are handled at the same time
        values, treedef = jax.tree_util.tree_flatten(state.params)

        # Init list of new parameters
        new_values = []

        for value, prior, trainable, forward in zip(
            values, priors, trainables, forwards
        ):
            if trainable:
                subkey, key = jax.random.split(key)
                new_value = forward(prior.sample(key))
                new_values.append(new_value)
            else:
                new_values.append(value)

        # Reconstruct the parameters dictionary with new values
        new_params = jax.tree_util.tree_unflatten(treedef, new_values)

        new_state = {"params": new_params} | opt

        return state.update(new_state)

    def transform(self, mode: str) -> "ModelState":
        if mode == "forward":
            transforms = self.params_forward_transforms
        elif mode == "backward":
            transforms = self.params_backward_transforms
        else:
            raise ValueError(
                f'mode should be either "forward" or "backward", got {mode}'
            )

        leaves, treedef = jax.tree_util.tree_flatten(self.params)
        new_leaves = [func(leaf) for func, leaf in zip(transforms, leaves)]
        new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
        return self.update({"params": new_params})
