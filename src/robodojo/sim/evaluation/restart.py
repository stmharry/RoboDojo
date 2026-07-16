"""PhysX restart policy and persisted-process recovery."""

from __future__ import annotations

import logging
import os
import sys

MAX_INPROC_RESTARTS = 3
logger = logging.getLogger(__name__)


def restart_or_exit(env, simulation_app, fatal_msg):
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0")) + 1
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as exc:
        logger.critical("persist_resume_manifest failed: %s", exc)
    logger.critical(
        "PhysX kernel failure detected: %s; persisted manifest. In-process restart attempt %s/%s.",
        fatal_msg,
        restart_count,
        MAX_INPROC_RESTARTS,
    )
    try:
        simulation_app.close()
    except Exception:
        pass
    if restart_count <= MAX_INPROC_RESTARTS:
        os.environ["ROBODOJO_FATAL_RESTART_COUNT"] = str(restart_count)
        logger.critical("os.execv self-restart with run_id=%s", os.environ.get("ROBODOJO_RUN_ID"))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, *sys.argv])
    logger.critical(
        "in-process restart cap reached (%s); exiting with rc=99 for shell-level retry.", MAX_INPROC_RESTARTS
    )
    raise SystemExit(99)


def exit_for_shell_restart(env, fatal_msg):
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0"))
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as exc:
        logger.critical("persist_resume_manifest failed: %s", exc)
    logger.critical("PhysX requested shell-level restart: %s; exiting with rc=99.", fatal_msg)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(99)
