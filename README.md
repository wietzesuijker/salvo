# salvo

Multi-cluster SLURM job runner for Mila and DRAC. Submit Python or shell from a decorator, route to the right account and partition, recover from OOM and preempt automatically.

## Install

    pip install pysalvo[mila,drac]
    salvo doctor

`doctor` checks topology detection, ssh alias hygiene, and the dataset manifest. Fix hints print under any WARN or FAIL.

## Submit a job

CLI:

    salvo submit --name hello --cpus 2 --mem 4G --time 30m -- echo hi

Dry-run (print the rendered sbatch without submitting):

    salvo submit --name hello --plan -- echo hi

Decorator (Python):

    from salvo import cluster

    @cluster.submit(gpus=1, cpus=8, mem="32G", time="2h",
                    on_oom=["bump_mem(1.5x, max=128G)", "fail"])
    def train(seed: int):
        ...

    handle = train.submit(seed=42)
    print(handle.job_id)

## Status, logs, cancel

    salvo status <job_id>
    salvo logs   <job_id> --stream stdout
    salvo logs   <job_id> --follow
    salvo cancel <job_id>

## Cluster detection

Order of precedence:

1. `SALVO_CLUSTER` env var
2. `CC_CLUSTER` env var (DRAC convention)
3. Hostname pattern match

Override per-call with `salvo submit --cluster <id>`.

Bundled presets: `mila`, `rorqual`, `narval`, `beluga`, `cedar`. Seven more DRAC clusters land in 0.2.

## Artifacts

Each submit writes `~/.salvo/runs/<job_id>/`:

- `spec.json`: validated `JobSpec`
- `sbatch.sh`: rendered script
- `events.jsonl`: append-only event stream
- `stdout`, `stderr`: once the job runs

## OOM policy DSL

    on_oom=[
        "bump_mem(1.5x, max=128G)",   # resubmit with bumped mem (CPU OOM)
        "escalate_partition",          # try a larger partition tier
        "fail",                        # terminal
    ]

Steps apply in order until one matches. `callback(...)` and `bump_gpus(...)` parse in 0.1 but execute starting 0.2.

## Contributing

The fastest contribution is a new cluster preset: one YAML file under `src/salvo/topology/presets/`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

MIT.
