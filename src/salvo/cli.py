"""salvo CLI: two narrow commands — doctor (self-check) and render (spec.yaml → sbatch)."""

from __future__ import annotations

import os
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from salvo.dispatch.account import pick_account
from salvo.dispatch.caps import CapsTracker
from salvo.dispatch.partition import pick_partition
from salvo.doctor import CheckStatus, run_doctor
from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec
from salvo.topology.detect import detect_cluster
from salvo.topology.loader import load_cluster

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
    artifact_dir: str = typer.Option(
        "<artifact_dir>", "--artifact-dir", help="path substituted into sbatch.sh"
    ),
) -> None:
    """Render a JobSpec (YAML) to sbatch text on stdout."""
    raw = yaml.safe_load(spec_path.read_text()) or {}
    spec = JobSpec.model_validate(raw)
    cid = cluster_id or detect_cluster()
    if cid is None:
        console.print("[red]could not detect cluster; pass --cluster or set SALVO_CLUSTER[/red]")
        raise typer.Exit(2)
    cluster = load_cluster(cid)
    caps = CapsTracker(cluster_id=cid, user=os.environ.get("USER", "unknown"))
    account = pick_account(spec, cluster, caps.snapshot())
    partition = pick_partition(spec, cluster, account=account)
    typer.echo(
        render_sbatch(
            spec,
            cluster=cluster,
            account=account,
            partition=partition,
            artifact_dir=artifact_dir,
        ),
        nl=False,
    )
