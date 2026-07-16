"""Policy communication lifecycle helpers."""

import logging

logger = logging.getLogger(__name__)


def close_model_client(env):
    try:
        model_client = getattr(env, "model_client", None)
        close = getattr(model_client, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        logger.warning("[main] failed to close model client: %s", exc)
