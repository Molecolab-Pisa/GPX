from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src import prng
from jax.typing import ArrayLike
from tabulate import tabulate

from .parameter import Parameter
from .utils import _is_numeric, _recursive_traverse_dict


@jax.tree_util.register_pytree_node_class
class ModelState:
    def __init__(
        self, kernel: Callable, params: Dict[str, Parameter], **kwargs: Any
    ) -> None:
        self.kernel = kernel
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

    def _params_priors(
        self, params: Dict[str, Parameter]
    ) -> List["Prior"]:  # noqa: F821
        return [p.prior for p in _recursive_traverse_dict(params)]

    @property
    def params_priors(self) -> List["Prior"]:  # noqa: F821
        return self._params_priors(self.params)

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

    def _params_value(self, params: Dict[str, Parameter]) -> List[Array]:
        return [p.value for p in _recursive_traverse_dict(params)]

    @property
    def params_value(self) -> List[Array]:
        return self._params_value(self.params)

    def _params_trainable(self, params: Dict[str, Parameter]) -> List[Array]:
        return [p.trainable for p in _recursive_traverse_dict(params)]

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
            self._params_structure,
            opt,
        )

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: ArrayLike) -> "ModelState":
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
            "forward",
            "backward",
            "prior",
            "type",
            "dtype",
            "shape",
            "value",
        ]

        def string_repr(p):
            return np.array2string(p.value, edgeitems=1, threshold=1)

        def get_info(p):
            v = p.value
            return (
                p.trainable,
                p.forward_transform.__name__,
                p.backward_transform.__name__,
                str(p.prior),
                type(v).__name__,
                v.dtype,
                v.shape,
                string_repr(p),
            )

        fields = [["kernel " + k] + list(get_info(p)) for k, p in kernel_params.items()]
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
        kernel_params = {
            "params:kernel_params:" + name: value.value
            for name, value in kernel_params.items()
        }
        params = {"params:" + name: value.value for name, value in params.items()}

        # Join
        params |= kernel_params

        # Get the auxiliary attributes
        for opt in self._register:
            params[opt] = getattr(self, opt)

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

        # auxiliary functions, to be clearer and less verbose
        # defined here as they only make sense within this method
        def is_param(p):
            return "params:" in p

        def is_kernel_param(p):
            return "params:kernel_params:" in p

        def patch_p_name(p):
            return p.replace("params:", "")

        def patch_kp_name(p):
            return p.replace("params:kernel_params:", "")

        # Load the new state values
        new_values = np.load(state_file, allow_pickle=True)
        new_values = {name: value for name, value in new_values.items()}

        # Init dicts for new parameters / optional variables
        new_params = {}
        new_params["kernel_params"] = {}
        new_opts = {}

        for name, value in new_values.items():
            if is_kernel_param(name):
                param = self.params["kernel_params"].get(patch_kp_name(name), None)
            elif is_param(name):
                param = self.params.get(patch_p_name(name), None)
            else:
                new_opts[name] = value
                continue

            # Check if the parameter is present in the model state
            # Since we can only update the value of the parameter,
            # it needs to be present. Otherwise, we cannot set the
            # transformation functions etc
            if param is None:
                raise ValueError(f"Cannot get parameter {param} from {self}")

            # Reconstruct the parameter
            _, aux = param.tree_flatten()
            param = param.tree_unflatten(aux, (value,))

            if is_kernel_param(name):
                new_params["kernel_params"][patch_kp_name(name)] = param
            elif is_param(name):
                new_params[patch_p_name(name)] = param
            else:
                raise RuntimeError("You should not be here.")

        update_dict = {"params": new_params} | new_opts

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
