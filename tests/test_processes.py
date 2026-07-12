import sys

from robodojo.core.processes import (
    format_command,
    free_port,
    run,
    start,
    terminate_process_group,
)

SMOKE_ENV = {
    "ROBODOJO_OPENARM_ZERO_ACTION": "1",
    "ROBODOJO_OPENARM_SMOKE_STEPS": "30",
}


def test_free_port_and_command_formatting():
    port = free_port()
    assert 0 < port < 65536
    rendered = format_command(["python", "-c", "print('hello world')"], {"GPU": "0"})
    assert rendered.startswith("GPU=0 python -c")
    assert "hello world" in rendered


def test_process_group_cleanup(tmp_path):
    process = start([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    terminate_process_group(process, grace=1)
    assert process.poll() is not None
    assert process.returncode in {-15, -9}


def test_process_start_uses_argv_without_shell_expansion(tmp_path):
    marker = tmp_path / "should-not-exist"
    process = start(
        [sys.executable, "-c", "import sys; assert sys.argv[1].startswith('$')", f"$({marker})"],
        cwd=tmp_path,
    )
    assert process.wait(timeout=5) == 0
    assert not marker.exists()


def test_process_helpers_strip_inherited_openarm_smoke_flags(monkeypatch, tmp_path):
    for name, value in SMOKE_ENV.items():
        monkeypatch.setenv(name, value)
    check = "import os, sys; sys.exit(any(name in os.environ for name in " + repr(tuple(SMOKE_ENV)) + "))"

    assert run([sys.executable, "-c", check], cwd=tmp_path) == 0
    process = start([sys.executable, "-c", check], cwd=tmp_path)
    assert process.wait(timeout=5) == 0


def test_process_helpers_allow_explicit_openarm_smoke_flags(monkeypatch, tmp_path):
    for name in SMOKE_ENV:
        monkeypatch.delenv(name, raising=False)
    check = (
        "import os, sys; expected = "
        + repr(SMOKE_ENV)
        + "; sys.exit(any(os.environ.get(k) != v for k, v in expected.items()))"
    )

    assert run([sys.executable, "-c", check], cwd=tmp_path, env=SMOKE_ENV) == 0
    process = start([sys.executable, "-c", check], cwd=tmp_path, env=SMOKE_ENV)
    assert process.wait(timeout=5) == 0
