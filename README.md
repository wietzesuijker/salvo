# salvo

A SLURM job is a shell script with a few `#SBATCH` headers. The catch: getting the headers right takes runs. Too little memory and the job dies after hours of compute. Too little time and it requeues. The wrong partition and it sits in the queue for a day. Each correction is a manual edit, push, resubmit.

Salvo replaces that loop with a typed `JobSpec`, a tiny DSL for what to do on OOM or preempt, a memory estimator that reads `sacct` history, and an `sbatch`-shelling submitter that records each hop. No SSH, no async.

## Install

    pip install pysalvo
    salvo doctor                                 # pre-flight checks
    salvo render spec.yaml --cluster mila        # JobSpec YAML to sbatch text

## Render

One source of truth for sbatch text; same inputs, same bytes every run.

```python
from salvo import JobSpec, render

spec = JobSpec(
    name="train", cmd=["python", "train.py"],
    gpus=1, cpus=8, mem="32G", time="2h",
    on_oom=["bump_mem(1.5x, max=128G)", "fail"],
)

sbatch_text = render(spec, cluster_id="mila", account="mila", partition="unkillable")
```

Leave `account`/`partition` off and salvo picks them via `salvo.dispatch` (login-node only, see below). `on_oom` is a list of DSL strings, validated when policy runs.

## Submit

`salvo.submit` shells out to `sbatch`, writes the rendered script plus `spec.json` and `cluster.json` into `~/.salvo/runs/<job_id>/`, and returns a `JobHandle`.

```python
from salvo import submit

handle = submit(spec)                                  # cluster auto-detected
handle.status().state                                  # PENDING | RUNNING | COMPLETED | OUT_OF_MEMORY | ...
for line in handle.logs(stream="stdout"):
    print(line)
handle.hops()                                          # list[Hop] read from hops.jsonl
handle.cancel()                                        # scancel
```

`submit(spec, hop="3/5", parent_artifact_dir=...)` is used by `salvo.runtime.epilog` for resubmit chains; user code rarely sets those directly.

## Decorator

`@cluster.submit(...)` wraps a module-level Python function as a remote-callable job. The decorated object is still callable locally; `.submit(**kwargs)` enqueues it on SLURM.

```python
from salvo import cluster

@cluster.submit(gpus=2, mem="32G", time="1h")
def train(seed: int, lr: float) -> None:
    ...

handle = train.submit(seed=0, lr=1e-3)
```

Signature is preserved through the wrapper (mypy `--strict`-friendly via `ParamSpec`). `.submit()` validates kwargs against the function signature, then JSON-round-trips them before constructing the `JobSpec`. Closures, lambdas, and `<locals>`-defined functions are rejected at decoration time (they cannot be re-imported on a worker).

## OOM policy

Declare the recovery strategy once instead of hand-editing `--mem` after every failure.

```python
from salvo.policy import parse, apply_oom, OomContext

steps = parse(["bump_mem(1.5x, max=128G)", "escalate_partition", "fail"])
new_spec, action = apply_oom(prev_spec, OomContext(kind="cpu", max_rss_mb=33_500))
```

DSL steps, applied in order: `bump_mem(<f>x, max=<size>)`, `escalate_partition`, `fail`.

## Memory estimator

Past `sacct` rows for the same `(script, commit, args)` triple become a P95 + safety estimate, so the next submit doesn't have to guess.

```python
from salvo.history import spec_key, estimate_mem, JobRecord

key = spec_key("train.sh", git_commit="cd1a0b4", program_args=("--seed", "0"))
records: list[JobRecord] = ...                         # cluv or your own cache supplies these
est = estimate_mem(records, safety=1.2, window=20, min_samples=3)

if est.mem_mb is not None:
    spec = spec.model_copy(update={"mem": f"{est.mem_mb}M"})
```

`spec_key` is a 12-byte blake2s; a code change resets history. `estimate_mem` returns `MemEstimate(mem_mb, confidence, n_samples, p95_mb, growth_slope_mb_per_run, rationale)`. `COMPLETED` jobs with `MaxRSS < 5%` of `ReqMem` are treated as degenerate sacct sampling and fall back to `ReqMem`. Salvo owns the math; the caller owns where the records live.

## Preempt

End-to-end resubmit. When `spec.on_preempt == "resubmit"`, `render` emits `#SBATCH --signal=SIGUSR1@90` and a bash `trap` that invokes `python -m salvo.runtime.epilog --artifact-dir "$SALVO_ARTIFACT_DIR" --reason preempt`. The epilog reads the parent's `spec.json` + `cluster.json`, bumps `SALVO_HOP`, calls `salvo.submit` for the child, and appends a `Hop` to `hops.jsonl`.

```python
from salvo.job.preempt import next_hop, should_resubmit, strip_account_suffix

new_hop, max_exceeded = next_hop("2/5")                # ("3/5", False)
strip_account_suffix("rrg-bengioy-ad_gpu")             # "rrg-bengioy-ad"
```

`max_hops` (default 5, capped at 20) bounds the chain.

## Topology and dispatch

Cluster knowledge as data: each cluster's accounts, partitions, capacity rules, and login constraints live in one YAML.

```python
from salvo.topology import load_preset, list_presets
from salvo.dispatch import pick_account, pick_partition, CapsTracker

cluster_obj = load_preset("mila")
snap = CapsTracker(cluster_id="mila", user="wietze").snapshot()
account = pick_account(spec, cluster_obj, snap)
partition = pick_partition(spec, cluster_obj, account)
```

Five presets ship: mila, rorqual, narval, beluga, cedar. Adding one is a single YAML. Dispatch is login-node only (shells out to `squeue`); library callers should pass `account`/`partition` to `render()` directly.

## Python entrypoint

The rendered sbatch script for a `PythonEntrypoint` cmd invokes:

    python -m salvo.runtime.entrypoint '{"target":"pkg.mod:fn","kwargs":{...}}'

Stdlib-only shim. Exit 2 on payload error; user exceptions propagate naturally so SLURM marks the job `FAILED`.

## Doctor

`salvo doctor` runs pre-flight checks (cluster detected, ssh alias not an FQDN, manifest fresh) and prints OK / WARN / FAIL with a one-line fix per check.

## With cluv

cluv handles SSH and `sbatch` from a laptop; salvo handles policy, memory, and on-cluster submit. Opt in via `pyproject.toml`:

```toml
[tool.cluv.retry]
on_oom = ["bump_mem(1.5x, max=128G)", "fail"]
max_hops = 5

[tool.cluv.estimate]
enabled = true
```

Two example wirings under [`examples/`](examples/) cover cluv (policy + render) and xgenius (policy-only).

## License

MIT. See [LICENSE](LICENSE).
