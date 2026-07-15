"""Repository and runtime path resolution."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict


def _is_repository_root(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    return 'name = "robodojo"' in pyproject.read_text(encoding="utf-8")


def discover_repository_root(explicit: str | Path | None = None) -> Path:
    """Resolve the checkout root without relying on package installation paths."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    elif root_from_env := os.environ.get("ROBODOJO_ROOT"):
        candidates.append(Path(root_from_env))
    else:
        candidates.extend((Path.cwd(), *Path.cwd().parents))
        source_checkout = Path(__file__).resolve().parents[3]
        candidates.extend((source_checkout, *source_checkout.parents))

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _is_repository_root(resolved):
            return resolved
    requested = f" {explicit!s}" if explicit is not None else ""
    raise RuntimeError(
        f"Could not locate a RoboDojo checkout{requested}. Run from the repository or set ROBODOJO_ROOT."
    )


class RepositoryPaths(BaseModel):
    """Canonical paths for a RoboDojo source checkout."""

    model_config = ConfigDict(frozen=True)

    root: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None) -> RepositoryPaths:
        return cls(root=discover_repository_root(root))

    @property
    def environment_configs(self) -> Path:
        return self.root / "configs"

    @property
    def environment_profiles(self) -> Path:
        return self.environment_configs / "environment"

    @property
    def task_configs(self) -> Path:
        return self.environment_configs / "task"

    @property
    def policy_profiles(self) -> Path:
        return self.environment_configs / "policies.yml"

    @property
    def task_protocols(self) -> Path:
        return self.environment_configs / "protocols.yml"

    @property
    def evaluation_recipes(self) -> Path:
        return self.environment_configs / "recipes.yml"

    @property
    def upstream_task_contracts(self) -> Path:
        return self.environment_configs / "reference" / "upstream_task_contracts.yml"

    @property
    def scene_profiles(self) -> Path:
        return self.environment_configs / "scene" / "profiles"

    @property
    def scene_components(self) -> Path:
        return self.environment_configs / "scene" / "components"

    @property
    def xpolicy_root(self) -> Path:
        return self.root / "XPolicyLab"

    @property
    def openarm_manifest(self) -> Path:
        return self.root / "configs" / "tooling" / "openarm.yml"

    @property
    def yam_manifest(self) -> Path:
        return self.root / "configs" / "tooling" / "yam.yml"

    @property
    def moonlake_office_manifest(self) -> Path:
        return self.root / "configs" / "tooling" / "moonlake_office.yml"

    @property
    def moonlake_packing_manifest(self) -> Path:
        return self.root / "configs" / "tooling" / "moonlake_packing.yml"
