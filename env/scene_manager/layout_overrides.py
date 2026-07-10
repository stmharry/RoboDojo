from collections.abc import Mapping
from copy import deepcopy

ALLOWED_LAYOUT_OVERRIDE_KEYS = frozenset({"Room", "Table", "Ground", "Background", "Light"})


def apply_fixture_overrides(layout: Mapping, overrides: Mapping) -> dict:
    forbidden = sorted(set(overrides) - ALLOWED_LAYOUT_OVERRIDE_KEYS)
    if forbidden:
        raise ValueError(f"layout_overrides contains non-fixture keys: {forbidden}")

    result = deepcopy(dict(layout))

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                target[key] = merge(dict(target[key]), value)
            else:
                target[key] = deepcopy(value)
        return target

    for key, value in overrides.items():
        if value is None:
            result[key] = None
        elif key in result and isinstance(result[key], Mapping):
            result[key] = merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result
