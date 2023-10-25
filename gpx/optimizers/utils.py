from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from jax import Array
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten
from jax.typing import ArrayLike

from ..parameters.parameter import Parameter, is_parameter

PyTreeDef = Any


def _get_trainable_fwd_or_truncate(p: Parameter) -> Optional[Callable]:
    return p.forward_transform if p.trainable else None


def get_trainable_forward_transforms(params: Dict[str, Parameter]) -> List[Callable]:
    """get the forward transform for trainable parameters

    If a parameter is trainable, gets the forward transformation
    If not, the tree is truncated with None and the remaining (trainable)
    leaves are extracted
    """
    return tree_leaves(
        tree_map(_get_trainable_fwd_or_truncate, params, is_leaf=is_parameter)
    )


def _get_trainable_bwd_or_truncate(p: Parameter) -> Optional[Callable]:
    return p.backward_transform if p.trainable else None


def get_trainable_backward_transforms(params: Dict[str, Parameter]) -> List[Callable]:
    """get the backward transform for trainable parameters

    If a parameter is trainable, gets the backward transformation
    If not, the tree is truncated with None and the remaining (trainable)
    leaves are extracted
    """
    return tree_leaves(
        tree_map(_get_trainable_bwd_or_truncate, params, is_leaf=is_parameter)
    )


def truncate_non_trainable_parameters(
    params: Dict[str, Parameter]
) -> Dict[str, Optional[Parameter]]:
    """creates a pytree where non-trainable parameters are truncated

    Non-trainable parameters are substituted with None, which
    indicates a termination of the tree branch
    """
    return tree_map(lambda p: p if p.trainable else None, params, is_leaf=is_parameter)


def ravel_backward_trainables(
    params: Dict[str, Parameter]
) -> Tuple[List[Array], PyTreeDef, Callable]:
    """flatten trainable parameters and apply backward

    This function takes a (generally nested) dictionary of parameters,
    extracts the trainable parameters only, applies the backward
    transformation, and flattens the unbound parameters in a 1D array.
    """
    # get the backward transformations for trainables
    bwd_fns = get_trainable_backward_transforms(params)
    # truncate and retain only trainables
    trainable_tree = truncate_non_trainable_parameters(params)
    # now xt only has the values of trainables
    # tdef builds a tree with only the trainables' values, and
    # None for non-trainables
    xt, tdef = tree_flatten(trainable_tree)
    # apply the backward transformation
    xt = tree_map(lambda v, fn: fn(v), xt, bwd_fns)
    # make the parameters 1D for the optimizer to work
    xt, unravel_fn = ravel_pytree(xt)
    return xt, tdef, unravel_fn


def unravel_forward_trainables(
    unravel_fn: Callable, tdef: PyTreeDef, params: Dict[str, Parameter]
) -> Callable:
    """unravel trainable parameters and apply forward

    This function generates a callable that can be used
    to unravel the 1D array of trainable parameters, apply
    the forward transformation to go in the original space, and
    reconstruct the full dictionary of parameters.

    Args:
        unravel_fn: function to unravel the 1D array of trainables
        tdef: tree definition for a tree of trainable parameters and None
              for non-trainable parameters
        params: original dictionary of parameters
    """
    fwd_fns = get_trainable_forward_transforms(params)

    def unravel_forward_func(xt: List[ArrayLike]) -> Dict[str, Parameter]:
        # restore the trainables' shape
        xt = unravel_fn(xt)
        # apply the forward transformation
        xt = tree_map(lambda v, fn: fn(v), xt, fwd_fns)
        # reconstruct the trainable pytree (non-trainables are still None)
        trainable_tree = tree_unflatten(tdef, xt)
        # map the trainable parameters into the original tree
        uparams = tree_map(
            lambda p0, p: p if p0.trainable else p0,
            params,
            trainable_tree,
            is_leaf=is_parameter,
        )
        return uparams

    return unravel_forward_func
