"""salvo CLI (typer). Phase 0 commands."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from salvo import submit as submit_fn
from salvo.dispatch.account import pick_account
from salvo.dispatch.caps import CapsTracker
from salvo.dispatch.partition import pick_partition
from salvo.doctor import CheckStatus, run_doctor
from salvo.job.handle import JobHandle
from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec
from salvo.topology.detect import detect_cluster
from salvo.topology.loader import load_cluster

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def doctor() -> None:
    """Self-check + fix hints."""
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
def submit(
    cmd: list[str] = typer.Argument(..., help="Command (use -- to separate)"),  # noqa: B008
    name: str = typer.Option("job", "--name"),
    gpus: int = typer.Option(0, "--gpus"),
    cpus: int = typer.Option(1, "--cpus"),
    mem: str = typer.Option("4G", "--mem"),
    time: str = typer.Option("1h", "--time"),
    partition: str | None = typer.Option(None, "--partition"),
    account: str | None = typer.Option(None, "--account"),
    container: str | None = typer.Option(None, "--container"),
    on_oom: list[str] | None = typer.Option(None, "--on-oom"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan", help="dry-run, print sbatch only"),
    cluster_id: str | None = typer.Option(None, "--cluster"),
) -> None:
    on_oom = on_oom or []
    spec = JobSpec(
        name=name,
        cmd=cmd,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        time=time,
        partition=partition,
        account=account,
        container=container,
        on_oom=on_oom,
    )
    if plan:
        cid = cluster_id or detect_cluster()
        if cid is None:
            console.print("[red]could not detect cluster[/red]")
            raise typer.Exit(2)
        cluster = load_cluster(cid)
        caps = CapsTracker(cluster_id=cid, user=os.environ.get("USER", "unknown"))
        acct = pick_account(spec, cluster, caps.snapshot())
        part = pick_partition(spec, cluster, account=acct)
        console.print(
            render_sbatch(
                spec,
                cluster=cluster,
                account=acct,
                partition=part,
                artifact_dir="<artifact_dir>",
            )
        )
        return
    h: JobHandle = submit_fn(spec, cluster_id=cluster_id)
    console.print(f"submitted: job_id={h.job_id} cluster={h.cluster}")


@app.command()
def status(job_id: str) -> None:
    salvo_root = Path(os.path.expanduser("~/.salvo"))
    artifact_dir = salvo_root / "runs" / job_id
    if not artifact_dir.exists():
        console.print(f"[red]no artifact dir for {job_id}[/red]")
        raise typer.Exit(1)
    spec_json = (artifact_dir / "spec.json").read_text()
    spec = JobSpec.model_validate_json(spec_json)
    h = JobHandle(job_id=job_id, cluster="", spec=spec, artifact_dir=artifact_dir)
    s = h.status()
    console.print(f"{job_id}: {s.state} (exit={s.exit_code}, elapsed={s.elapsed_min}m)")


@app.command()
def logs(
    job_id: str,
    stream: str = typer.Option("stdout", "--stream"),
    follow: bool = typer.Option(False, "--follow"),
) -> None:
    salvo_root = Path(os.path.expanduser("~/.salvo"))
    artifact_dir = salvo_root / "runs" / job_id
    spec = JobSpec.model_validate_json((artifact_dir / "spec.json").read_text())
    h = JobHandle(job_id=job_id, cluster="", spec=spec, artifact_dir=artifact_dir)
    for line in h.logs(stream=stream, follow=follow):  # type: ignore[arg-type]
        console.print(line)


@app.command()
def cancel(job_id: str) -> None:
    salvo_root = Path(os.path.expanduser("~/.salvo"))
    artifact_dir = salvo_root / "runs" / job_id
    spec = JobSpec.model_validate_json((artifact_dir / "spec.json").read_text())
    h = JobHandle(job_id=job_id, cluster="", spec=spec, artifact_dir=artifact_dir)
    h.cancel()
    console.print(f"cancelled {job_id}")
