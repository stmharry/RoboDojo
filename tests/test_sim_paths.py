from robodojo.sim.environment.global_configs import ASSETS_PATH
from robodojo.sim.utils.path import resolve_path


def test_asset_placeholder_accepts_canonical_and_saved_layout_spellings():
    suffix = "/Material/Garment/train/example.usd"
    assert resolve_path(f"$RoboDojo_ASSETS{suffix}") == f"{ASSETS_PATH}{suffix}"
    assert resolve_path(f"$Robodojo_ASSETS{suffix}") == f"{ASSETS_PATH}{suffix}"
