from __future__ import annotations

from typing import Dict, Generator

from jax.typing import ArrayLike


def _recursive_traverse_dict(dictionary: Dict) -> Generator[ArrayLike, None, None]:
    for key in dictionary.keys():
        value = dictionary[key]
        if isinstance(value, dict):
            yield from _recursive_traverse_dict(value)
        else:
            yield value
