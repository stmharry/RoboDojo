"""Typed experiment catalog loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from robodojo.core.models.experiment import EvaluationRecipeCatalog, PolicyProfileCatalog, TaskProtocolCatalog
from robodojo.core.models.policy import (
    PolicyEvaluationContract,
    PolicyEvaluationContractCatalog,
    PolicyProfileDocument,
)
from robodojo.core.paths import RepositoryPaths


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"experiment catalog not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_policy_catalog(paths: RepositoryPaths) -> PolicyProfileCatalog:
    return PolicyProfileCatalog.model_validate(load_yaml(paths.policy_profiles))


def load_policy_evaluation_contract(
    paths: RepositoryPaths,
    policy: PolicyProfileDocument,
) -> tuple[PolicyEvaluationContract, Path, str]:
    policy_dir = (paths.root / policy.policy_dir).resolve()
    if not policy_dir.is_relative_to(paths.xpolicy_root.resolve()):
        raise ValueError(f"tracked policy directory must stay below XPolicyLab: {policy.policy_dir}")
    descriptor_path = policy_dir / "eval_contracts.yml"
    catalog = PolicyEvaluationContractCatalog.model_validate(load_yaml(descriptor_path))
    try:
        descriptor = catalog.profiles[policy.checkpoint]
    except KeyError as exc:
        raise ValueError(
            f"policy checkpoint {policy.checkpoint!r} has no entry in {descriptor_path.relative_to(paths.root)}"
        ) from exc
    encoded = json.dumps(descriptor.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    return descriptor, descriptor_path, hashlib.sha256(encoded).hexdigest()


def load_protocol_catalog(paths: RepositoryPaths) -> TaskProtocolCatalog:
    return TaskProtocolCatalog.model_validate(load_yaml(paths.task_protocols))


def load_recipe_catalog(paths: RepositoryPaths) -> EvaluationRecipeCatalog:
    return EvaluationRecipeCatalog.model_validate(load_yaml(paths.evaluation_recipes))
