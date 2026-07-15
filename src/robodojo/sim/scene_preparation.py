"""Named, lazy scene-asset preparation hooks."""

from __future__ import annotations

from collections.abc import Callable

from robodojo.core.profiles import SceneProfile


def _yam_short_sleeve_garment() -> None:
    from robodojo.workflows.assets_yam_scene import prepare_yam_short_sleeve_garment

    prepare_yam_short_sleeve_garment()


_PREPARERS: dict[str, Callable[[], None]] = {
    "yam_short_sleeve_garment": _yam_short_sleeve_garment,
}


def validate_scene_preparers(profile: SceneProfile) -> None:
    unknown = sorted(
        {
            name
            for preparers in profile.document.task_asset_preparers.values()
            for name in preparers
            if name not in _PREPARERS
        }
    )
    if unknown:
        raise ValueError(f"scene profile {profile.name} references unknown asset preparers: {unknown}")


def prepare_scene_assets(profile: SceneProfile, task_name: str) -> None:
    validate_scene_preparers(profile)
    for preparer_name in profile.document.task_asset_preparers.get(task_name, ()):
        _PREPARERS[preparer_name]()
