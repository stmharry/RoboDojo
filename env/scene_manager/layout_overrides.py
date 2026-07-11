from collections.abc import Mapping
from copy import deepcopy

ALLOWED_LAYOUT_OVERRIDE_KEYS = frozenset(
    {"Room", "Table", "Ground", "Background", "Light", "remove_fixtures"}
)
ALLOWED_FIXTURE_REMOVALS = frozenset({"Geometry.camera_stand"})


def apply_fixture_overrides(layout: Mapping, overrides: Mapping) -> dict:
    forbidden = sorted(set(overrides) - ALLOWED_LAYOUT_OVERRIDE_KEYS)
    if forbidden:
        raise ValueError(f"layout_overrides contains non-fixture keys: {forbidden}")

    result = deepcopy(dict(layout))

    removals = overrides.get("remove_fixtures", [])
    if not isinstance(removals, (list, tuple)):
        raise ValueError("layout_overrides.remove_fixtures must be a list")
    forbidden_removals = sorted(set(removals) - ALLOWED_FIXTURE_REMOVALS)
    if forbidden_removals:
        raise ValueError(f"layout_overrides contains forbidden fixture removals: {forbidden_removals}")
    for removal in removals:
        object_type, category = removal.split(".", maxsplit=1)
        categories = result.get(object_type)
        if isinstance(categories, Mapping):
            categories = dict(categories)
            categories.pop(category, None)
            if categories:
                result[object_type] = categories
            else:
                result.pop(object_type, None)

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                target[key] = merge(dict(target[key]), value)
            else:
                target[key] = deepcopy(value)
        return target

    for key, value in overrides.items():
        if key == "remove_fixtures":
            continue
        if value is None:
            result[key] = None
        elif key in result and isinstance(result[key], Mapping):
            result[key] = merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result
