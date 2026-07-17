import io
import json
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.experiments.presentation import recipe_rows
from robodojo.core.paths import RepositoryPaths
from robodojo.workflows.recipe_inventory import print_recipe_table

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()
LONG_RECIPE = "molmoact2-bimanual_yam-moonlake_office-moonlake_office_general_pickup"


def _rows():
    return recipe_rows(RepositoryPaths.resolve(ROOT))


def test_recipe_table_groups_deterministically_without_tsv_output():
    result = RUNNER.invoke(app, ["catalog", "recipes", "--format", "table", "--root", str(ROOT)])

    assert result.exit_code == 0
    assert "Tracked evaluation recipes (26)" in result.stdout
    assert "Policy: molmoact2_bimanual_yam" in result.stdout
    assert "Environment: bimanual_yam_moonlake_office" in result.stdout
    assert "Scene: moonlake_office" in result.stdout
    assert "moonlake_office_general_pickup" in result.stdout
    assert "Base task" in result.stdout
    assert "general_pickup" in result.stdout
    assert "\t" not in result.stdout

    headings = [
        "Policy: lerobot_pi05_openarm",
        "Policy: molmoact2_bimanual_yam",
        "Policy: pi05_arx_x5",
        "Policy: pi05_bimanual_yam",
        "Policy: pi05_bimanual_yam_pickup",
        "Policy: smolvla_arx_x5",
    ]
    assert [result.stdout.index(heading) for heading in headings] == sorted(
        result.stdout.index(heading) for heading in headings
    )


def test_recipe_table_sorts_rows_and_shows_the_complete_mapping():
    stream = io.StringIO()
    console = Console(file=stream, width=200, color_system=None, force_terminal=False)
    moonlake_rows = [
        row
        for row in reversed(_rows())
        if row["policy"] == "molmoact2_bimanual_yam" and row["scene"] == "moonlake_office"
    ]

    print_recipe_table(moonlake_rows, console=console)

    rendered = stream.getvalue()
    recipe_names = sorted(row["recipe"] for row in moonlake_rows)
    assert [rendered.index(name) for name in recipe_names] == sorted(rendered.index(name) for name in recipe_names)
    recipe_line = next(line for line in rendered.splitlines() if LONG_RECIPE in line)
    assert "moonlake_office_general_pickup" in recipe_line
    assert recipe_line.count("general_pickup") == 3


def test_narrow_recipe_table_folds_without_truncating_recipe_names():
    stream = io.StringIO()
    console = Console(file=stream, width=48, color_system=None, force_terminal=False)
    row = next(row for row in _rows() if row["recipe"] == LONG_RECIPE)
    isolated_row = {**row, "task_protocol": "", "task": ""}

    print_recipe_table([isolated_row], console=console)

    rendered = stream.getvalue()
    assert LONG_RECIPE in "".join(rendered.split())
    assert "…" not in rendered


def test_plain_and_json_recipe_formats_remain_machine_compatible():
    rows = _rows()
    default = RUNNER.invoke(app, ["catalog", "recipes", "--root", str(ROOT)])
    plain = RUNNER.invoke(app, ["catalog", "recipes", "--format", "plain", "--root", str(ROOT)])
    structured = RUNNER.invoke(app, ["catalog", "recipes", "--format", "json", "--root", str(ROOT)])
    checked = RUNNER.invoke(
        app,
        ["catalog", "recipes", "--format", "json", "--check", "--root", str(ROOT)],
    )

    expected_plain = [
        "\t".join(row[field] for field in ("recipe", "policy", "environment", "scene", "task_protocol", "task"))
        for row in rows
    ]
    assert all("layout" not in row for row in rows)
    assert default.exit_code == 0
    assert default.stdout == plain.stdout
    assert plain.exit_code == 0
    assert plain.stdout.splitlines() == expected_plain
    assert structured.exit_code == 0
    assert json.loads(structured.stdout) == rows
    assert checked.exit_code == 0
    assert json.loads(checked.stdout) == rows
