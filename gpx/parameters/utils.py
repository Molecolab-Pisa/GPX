def _recursive_traverse_dict(dictionary):
    for key in dictionary.keys():
        value = dictionary[key]
        if isinstance(value, dict):
            yield from _recursive_traverse_dict(value)
        else:
            yield value
