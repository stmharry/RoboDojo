#!/usr/bin/env bash
# Run the requested command from the project root. The image PATH already puts
# the locked uv environment first.
set -e

cd /workspace/RoboDojo

exec "$@"
