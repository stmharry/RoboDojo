"""Typed builder registry replacing asset-name branching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from robodojo.core.paths import RepositoryPaths

AssetKind = Literal["robot", "fixture"]


@dataclass(frozen=True)
class AssetBuilder:
    name: str
    kind: AssetKind
    build: Callable[[RepositoryPaths], int]
    validate: Callable[[RepositoryPaths], str | None]


class AssetBuilderRegistry:
    def __init__(self, builders: tuple[AssetBuilder, ...]):
        self._builders = {(builder.kind, builder.name): builder for builder in builders}
        if len(self._builders) != len(builders):
            raise ValueError("asset builder registrations must be unique")

    def get(self, kind: AssetKind, name: str) -> AssetBuilder:
        try:
            return self._builders[(kind, name)]
        except KeyError as exc:
            raise ValueError(f"unsupported generated {kind} asset: {name}") from exc
