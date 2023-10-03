from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from jax import Array
from jax.typing import ArrayLike


def _check_same_shape(a: ArrayLike, b: ArrayLike, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{a_name}.shape={a.shape} and {b_name}.shape={b.shape} do not match."
        )


def _check_same_dtype(a: ArrayLike, b: ArrayLike, a_name: str, b_name: str) -> None:
    if a.dtype != b.dtype:
        raise ValueError(
            f"{a_name}.dtype={a.dtype} and {b_name}.dtype={b.dtype} do not match."
        )


def _is_numeric(value: Any) -> bool:
    """check if value is an integer, unsigned integer, or a float

    Returns True if value is either an integer, an unsigned
    integer, or a float, False otherwise.
    Works both with python numbers and numpy/jax arrays. If the latter,
    the array dtype is checked instead of the object type.
    """
    vtype = value.dtype if isinstance(value, (np.ndarray, Array)) else type(value)
    is_unsigned_integer = (
        np.issubdtype(vtype, np.uint8)
        or np.issubdtype(vtype, np.uint16)
        or np.issubdtype(vtype, np.uint32)
        or np.issubdtype(vtype, np.uint64)
    )
    is_integer = (
        np.issubdtype(vtype, int)
        or np.issubdtype(vtype, np.int8)
        or np.issubdtype(vtype, np.int16)
        or np.issubdtype(vtype, np.int32)
        or np.issubdtype(vtype, np.int64)
    )
    is_float = (
        np.issubdtype(vtype, float)
        or np.issubdtype(vtype, np.float16)
        or np.issubdtype(vtype, np.float32)
        or np.issubdtype(vtype, np.float64)
    )
    return is_unsigned_integer or is_integer or is_float


def _flatten_dict(
    dictionary, flattened=None, starting_key=None, sep=":", map_value=None
):
    # instantiate the flattened dictionary
    if flattened is None:
        flattened = {}

    # We want to be compatible with a final, empty
    # dictionary. We use the name 'VOID' to indicate
    # the empty dictionary
    if len(dictionary.keys()) == 0:
        k = f"{starting_key}{sep}NULL" if starting_key else "NULL"
        flattened[k] = None

    else:
        for k, v in dictionary.items():
            k = f"{starting_key}{sep}{k}" if starting_key else k
            if isinstance(v, dict):
                _flatten_dict(
                    dictionary=v,
                    flattened=flattened,
                    starting_key=k,
                    sep=sep,
                    map_value=map_value,
                )
                continue

            flattened[k] = map_value(v) if map_value is not None else v

    return flattened


def _unflatten_dict(dictionary, sep=":"):
    # instantiate the unflattened dictionary
    unflattened = {}

    for k in dictionary.keys():
        # Try to retrieve a dictionary key but do not
        # stop iterating if you're unable to get that
        try:
            v = dictionary[k]
        except ValueError:
            warnings.warn(f"Unable to retrieve key={k}", stacklevel=2)
            continue

        # start from top
        cur_dict = unflattened

        # split into subkeys
        keys = k.split(sep)

        for subk in keys:
            # we are at the last key, so we assign the value now
            if subk is keys[-1]:
                # NULL identifies an empty dictionary, so we
                # do nothing
                if subk == "NULL":
                    pass
                else:
                    cur_dict[subk] = v
            # Avoid overwriting on existing keys
            elif subk in cur_dict:
                cur_dict = cur_dict[subk]
            # Key not present, so we instantiate a new dict
            else:
                cur_dict[subk] = {}
                cur_dict = cur_dict[subk]

    return unflattened
