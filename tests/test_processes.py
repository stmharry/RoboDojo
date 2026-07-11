import sys

from robodojo.core.processes import format_command, free_port, start, terminate_process_group


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
