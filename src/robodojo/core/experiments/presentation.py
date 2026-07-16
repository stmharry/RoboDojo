"""User-facing projections of resolved experiments."""

from robodojo.core.experiments.validation import validate_experiment_catalogs
from robodojo.core.paths import RepositoryPaths


def recipe_rows(paths: RepositoryPaths) -> list[dict[str, str]]:
    return [
        {
            "recipe": experiment.name or "manual",
            "policy": experiment.policy_name,
            "environment": experiment.environment.name,
            "scene": experiment.scene.name,
            "task_protocol": experiment.task_protocol,
            "task": experiment.protocol.task,
            "reference_match": experiment.policy_reference_match,
        }
        for experiment in validate_experiment_catalogs(paths)
    ]
