from typing import Any, Dict


def get_flattened_dict(d: dict, concat_str: str = "_") -> Dict[str, Any]:
    """Flattens a nested dict into a single level dict with concatenated keys."""
    flat_dict = {}

    def flatten(d: dict, parent_key: str = "") -> None:
        for key, val in d.items():
            if parent_key:
                key = parent_key + concat_str + key
            if not isinstance(val, dict):
                assert key not in flat_dict, f'Key "{key}" already exists in dict!!!'
                flat_dict[key] = val
            else:
                flatten(val, key)

    flatten(d)
    return flat_dict
