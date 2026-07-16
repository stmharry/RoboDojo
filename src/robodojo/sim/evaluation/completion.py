"""Evaluation completion checks independent of simulator initialization."""


class IncompleteEvaluationError(RuntimeError):
    """Raised when the requested episode count cannot be collected."""


def ensure_evaluation_complete(*, eval_time: int, requested: int) -> None:
    """Reject partial evaluations that exhausted their available layouts."""

    if eval_time < requested:
        raise IncompleteEvaluationError(
            "evaluation exhausted available layouts before reaching the requested "
            f"episode count: completed={eval_time}, requested={requested}"
        )
