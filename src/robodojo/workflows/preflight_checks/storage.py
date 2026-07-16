"""Read-only experiment validation."""

from __future__ import annotations

import shlex
import shutil

from robodojo.core.models.reports import (
    PreflightCheck,
)
from robodojo.core.models.requests import (
    PreflightRequest,
)
from robodojo.core.storage import s3_uri
from robodojo.workflows.preflight_checks.reporting import _check

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "setup", "--only", "root"]
)


def _publication_check(request: PreflightRequest) -> PreflightCheck:
    if not request.publish:
        return _check("publication", "PASS", "publication disabled")
    remote = s3_uri()
    if remote is None or not remote.startswith("s3://"):
        return _check(
            "publication",
            "FAIL",
            "ROBODOJO_S3_URI is not a dedicated s3:// prefix",
            "export ROBODOJO_S3_URI in the process environment",
        )
    aws = shutil.which("aws")
    if aws is None:
        return _check("publication", "FAIL", "AWS CLI is unavailable", "install and configure the AWS CLI")
    return _check("publication", "PASS", f"AWS CLI and destination {remote} are configured")
