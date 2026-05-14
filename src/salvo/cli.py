"""salvo CLI: two narrow commands — doctor (self-check) and render (spec.yaml → sbatch)."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from salvo import render as render_spec
from salvo.doctor import CheckStatus, run_doctor
from salvo.errors import SalvoError
from salvo.job.spec import JobSpec

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="salvo: render sbatch from JobSpec, plus a self-check. Transport lives elsewhere.",
)
console = Console()


@app.command()
def doctor() -> None:
    """Self-check: topology detection, ssh alias hygiene, manifest freshness."""
    results = run_doctor()
    table = Table(title="salvo doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Message")
    fail = False
    for r in results:
        sty = {"OK": "green", "WARN": "yellow", "FAIL": "red"}[r.status.value]
        table.add_row(r.name, f"[{sty}]{r.status.value}[/{sty}]", r.message)
        if r.status is CheckStatus.FAIL:
            fail = True
    console.print(table)
    for r in results:
        if r.fix:
            console.print(f"[dim]fix:[/dim] {r.fix}")
    raise typer.Exit(1 if fail else 0)


@app.command()
def render(
    spec_path: Path = typer.Argument(  # noqa: B008
        ..., help="YAML file with JobSpec fields"
    ),
    cluster_id: str | None = typer.Option(
        None, "--cluster", help="cluster id (defaults to detect)"
    ),
    account: str | None = typer.Option(
        None, "--account", help="override picked account; skips squeue when both set"
    ),
    partition: str | None = typer.Option(
        None, "--partition", help="override picked partition; skips squeue when both set"
    ),
    artifact_dir: str = typer.Option(
        "<artifact_dir>", "--artifact-dir", help="path substituted into sbatch.sh"
    ),
) -> None:
    """Render a JobSpec (YAML) to sbatch text on stdout.

    Pure when --account and --partition are both supplied (or pinned on the
    spec). Otherwise shells out to squeue for capacity-aware picking, which
    only works on a SLURM login node.
    """
    raw = yaml.safe_load(spec_path.read_text()) or {}
    spec = JobSpec.model_validate(raw)
    try:
        text = render_spec(
            spec,
            cluster_id=cluster_id,
            account=account,
            partition=partition,
            artifact_dir=artifact_dir,
        )
    except SalvoError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    typer.echo(text, nl=False)
