# salvo

Policy and render library for SLURM. salvo turns a `JobSpec` into byte-stable sbatch text and decides what to do when a job hits OOM or gets preempted. It does not move code to the cluster, does not open SSH connections, and is not a runner. For transport, pair it with [cluv](https://github.com/mila-iqia/cluv) or invoke `sbatch` directly.

## Install

    pip install pysalvo

Two CLI commands ship for diagnostics and one-shot rendering. Everything else is meant to be imported.

    salvo doctor                                 # topology, ssh alias hygiene, manifest freshness
    salvo render spec.yaml --cluster mila        # JobSpec YAML to sbatch text on stdout

## Render

```python
from salvo import JobSpec, render

spec = JobSpec(
    name="train",
    cmd=["python", "train.py"],
    gpus=1, cpus=8, mem="32G", time="2h",
    on_oom=["bump_mem(1.5x, max=128G)", "fail"],
)

sbatch_text = render(
    spec,
    cluster_id="mila",
    account="mila",
    partition="unkillable",
)  # str, byte-stable, no side effects
```

Same JobSpec, same cluster, same inputs, same bytes every run. Useful when you need to audit what was actually submitted six months later.

If you leave `account` and `partition` off, salvo picks them via `salvo.dispatch` by shelling out to `squeue` for live capacity. That path only works on a SLURM login node; library callers (cluv, xgenius) should pass both explicitly to keep `render()` pure.

## OOM policy

```python
from salvo.policy import parse, apply_oom, OomContext

steps = parse(["bump_mem(1.5x, max=128G)", "escalate_partition", "fail"])

new_spec, action = apply_oom(prev_spec, OomContext(class_="cpu", max_rss_mb=33_500))
# new_spec is a fresh JobSpec with bumped mem, or None if the policy says fail
```

The DSL is intentionally small. Steps run in order until one applies:

- `bump_mem(<factor>x, max=<size>)` — multiplicative bump, capped
- `escalate_partition` — clear partition so the next render picks a larger tier
- `fail` — terminal
- `bump_gpus(...)`, `callback(...)` — parse today, execute in a later release

## Cluster topology

Five presets ship in `salvo/topology/presets/`: `mila`, `rorqual`, `narval`, `beluga`, `cedar`. More DRAC clusters land as YAMLs are contributed. Each YAML lists accounts, partitions, and capacity rules.

```python
from salvo.topology import load_preset, list_presets

list_presets()                      # ['beluga', 'cedar', 'mila', 'narval', 'rorqual']
cluster = load_preset("mila")       # ClusterTopology
```

Contributing a new cluster is one YAML file. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Decorator (optional, Python-native)

If you'd rather write a function than a YAML spec:

```python
from salvo import cluster

@cluster.submit(gpus=1, cpus=8, mem="32G", time="2h",
                on_oom=["bump_mem(1.5x, max=128G)", "fail"])
def train(seed: int):
    ...

handle = train.submit(seed=42)
```

`train.submit(...)` runs the local-sbatch convenience path: it renders, calls `sbatch`, and returns a handle. Skip the decorator if you're embedding salvo inside another tool's submit flow; just call `render()` and let that tool do the submission.

## Use with cluv

cluv handles SSH, code sync, and the `sbatch` call. salvo handles the policy. The natural composition is cluv importing salvo's policy library when a user opts in:

```toml
# in your project's pyproject.toml
[tool.cluv.retry]
on_oom = ["bump_mem(1.5x, max=128G)", "fail"]
max_hops = 5
```

This is a proposal, not a shipped feature in cluv. An upstream issue is in draft.

## Use with anything that calls sbatch

salvo has no SSH or async dependencies. It is a pydantic + stdlib library. Any tool that runs `sbatch` can import `salvo.render`, `salvo.policy.apply_oom`, and `salvo.topology.load_preset` independently. Two runnable examples under [`examples/`](examples/) show the wiring end-to-end: [`cluv_integration.py`](examples/cluv_integration.py) (policy + render) and [`xgenius_integration.py`](examples/xgenius_integration.py) (policy-only, when the host tool keeps its own renderer).

## License

MIT. See [LICENSE](LICENSE).
