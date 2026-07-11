"""Constrained post-layout appearance overrides.

These overlays deliberately cannot mutate task objects, fixture geometry, physics,
or scoring inputs.  They are applied after a saved evaluation layout is loaded.
"""

from collections.abc import Mapping
from copy import deepcopy

ALLOWED_APPEARANCE_KEYS = frozenset({"Room", "Table", "Ground", "Background", "Light"})


def apply_appearance_overrides(layout: Mapping, overrides: Mapping) -> dict:
    forbidden = sorted(set(overrides) - ALLOWED_APPEARANCE_KEYS)
    if forbidden:
        raise ValueError(f"appearance_overrides contains non-appearance keys: {forbidden}")

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                target[key] = merge(dict(target[key]), value)
            else:
                target[key] = deepcopy(value)
        return target

    result = deepcopy(dict(layout))
    for key, value in overrides.items():
        if value is None:
            result[key] = None
        elif key in result and isinstance(result[key], Mapping):
            result[key] = merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result
