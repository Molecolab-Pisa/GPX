from typing import Any, Dict, Generator

Array = Any


def _recursive_traverse_dict(dictionary: Dict) -> Generator[Array]:
    for key in dictionary.keys():
        value = dictionary[key]
        if isinstance(value, dict):
            yield from _recursive_traverse_dict(value)
        else:
            yield value
