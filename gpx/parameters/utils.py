from __future__ import annotations

from typing import Any, Dict, Generator

import numpy as np
from jax import Array
from jax.typing import ArrayLike


def _recursive_traverse_dict(dictionary: Dict) -> Generator[ArrayLike, None, None]:
    for key in dictionary.keys():
        value = dictionary[key]
        if isinstance(value, dict):
            yield from _recursive_traverse_dict(value)
        else:
            yield value


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
