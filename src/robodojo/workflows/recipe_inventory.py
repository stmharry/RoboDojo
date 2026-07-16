"""Human-readable presentation for the typed evaluation recipe catalog."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


def _group_heading(policy: str, environment: str, scene: str, reference_match: str) -> Text:
    heading = Text()
    fields = (
        ("Policy", policy),
        ("Environment", environment),
        ("Scene", scene),
        ("Training fit", reference_match),
    )
    for index, (label, value) in enumerate(fields):
        if index:
            heading.append("  •  ", style="dim")
        heading.append(f"{label}: ", style="bold cyan")
        heading.append(value, style="bold")
    return heading


def print_recipe_table(rows: Sequence[Mapping[str, str]], *, console: Console | None = None) -> None:
    """Print complete recipe identities in compact, deterministic groups."""

    output = console or Console()
    groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in sorted(
        rows,
        key=lambda item: (
            item["policy"],
            item["environment"],
            item["scene"],
            item["recipe"],
        ),
    ):
        groups[(row["policy"], row["environment"], row["scene"])].append(row)

    output.print(f"Tracked evaluation recipes ({len(rows)})", style="bold")
    for index, ((policy, environment, scene), recipes) in enumerate(sorted(groups.items())):
        if index:
            output.print()
        output.print(_group_heading(policy, environment, scene, recipes[0]["reference_match"]))
        table = Table(box=box.SIMPLE, expand=True, pad_edge=False, show_edge=False)
        table.add_column("Recipe", ratio=5, overflow="fold")
        table.add_column("Task protocol", min_width=32, overflow="fold", no_wrap=True)
        table.add_column("Base task", ratio=2, overflow="fold")
        for row in recipes:
            table.add_row(row["recipe"], row["task_protocol"], row["task"])
        output.print(table)
