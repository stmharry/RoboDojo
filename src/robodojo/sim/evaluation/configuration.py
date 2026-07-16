"""Evaluation and policy-transport configuration assembly."""

from __future__ import annotations

from copy import deepcopy

from robodojo.core.artifacts.results import ARTIFACT_SCHEMA_VERSION


def build_evaluation_config(
    *,
    environment,
    scene,
    layout_set,
    environment_assets,
    scene_assets,
    runtime_experiment,
    task: str,
    task_protocol: str,
    episode_horizon: int,
    evaluation_episodes: int,
    recipe: str,
    experiment_hash: str,
    policy_name: str,
    policy_profile: str,
    seed: int,
    additional_info: str,
    device_id: int,
    num_envs: int,
    physx_monitor_enabled: bool,
    robodojo_revision: str,
    xpolicylab_revision: str,
    assets_root,
) -> dict:
    config = deepcopy(environment.payload)
    config.update(
        {
            "environment": environment.name,
            "environment_profile_hash": environment.identity_hash,
            "environment_variant": (
                environment.document.variant.model_dump(mode="json", exclude_none=True)
                if environment.document.variant is not None
                else None
            ),
            "environment_asset_hash": environment_assets.identity_hash,
            "environment_asset_builds": list(environment.document.asset_builds),
            "environment_asset_identities": list(environment_assets.artifacts),
            "embodiment": environment.embodiment,
            "scene": scene.name,
            "scene_component": scene.document.component,
            "scene_profile_hash": scene.identity_hash,
            "layout_set": scene.document.layout_set,
            "layout_source": scene.document.layout_source,
            "layout_set_hash": layout_set.identity_hash,
            "scene_asset_hash": scene_assets.identity_hash,
            "scene_asset_builds": list(
                dict.fromkeys((*scene.document.asset_builds, *scene.document.task_asset_builds.get(task, ())))
            ),
            "scene_asset_identities": [
                {
                    "destination": artifact.destination_root.relative_to(assets_root).as_posix(),
                    "derivation_hash": artifact.derivation_hash,
                    "manifest_hash": artifact.manifest_hash,
                }
                for artifact in scene_assets.artifacts
            ],
            "task": task,
            "task_protocol": task_protocol,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "episode_horizon": episode_horizon,
            "evaluation_episodes": evaluation_episodes,
            "recipe": recipe,
            "experiment_hash": experiment_hash,
            "policy_profile": policy_profile,
            "policy_descriptor_hash": (
                runtime_experiment.policy_descriptor_hash if runtime_experiment is not None else None
            ),
            "policy_reference_match": (
                runtime_experiment.policy_reference_match if runtime_experiment is not None else "unspecified"
            ),
            "policy_checkpoint": runtime_experiment.policy.checkpoint if runtime_experiment is not None else None,
            "policy_execution": (
                runtime_experiment.policy_descriptor.execution.model_dump(mode="json")
                if runtime_experiment is not None
                else None
            ),
            "policy_training": (
                runtime_experiment.policy_descriptor.training.model_dump(mode="json")
                if runtime_experiment is not None
                else None
            ),
            "policy_adapter": (
                runtime_experiment.policy_descriptor.adapter.model_dump(mode="json")
                if runtime_experiment is not None
                else None
            ),
            "robodojo_revision": robodojo_revision,
            "xpolicylab_revision": xpolicylab_revision,
            "num_envs": num_envs,
            "device_id": device_id,
            "policy_name": policy_name,
            "additional_info": additional_info,
            "seed": seed,
            "physx_monitor_enabled": physx_monitor_enabled,
        }
    )
    return config


def build_deploy_config(
    *, policy_name: str, host: str, port: int, transport: str, server_url: str, run_id: str, task_protocol: str
) -> dict:
    """Keep XPolicyLab's `protocol` vocabulary isolated at its boundary."""

    return {
        "policy_name": policy_name,
        "port": port,
        "host": host,
        "protocol": transport,
        "policy_server_url": server_url or f"ws://{host}:{port}",
        "evaluation_id": run_id,
        "trial_id": f"{task_protocol}-{run_id}",
        "action_case_id": f"{task_protocol}_case",
        "repeat_index": None,
    }
