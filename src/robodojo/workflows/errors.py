"""Expected workflow failures that command adapters render without tracebacks."""


class WorkflowError(RuntimeError):
    """Base class for a user-actionable workflow failure."""


class StorageError(WorkflowError):
    """A durable-storage operation could not be completed safely."""


class ResultsError(WorkflowError):
    """Evaluation results could not be selected or reported unambiguously."""
