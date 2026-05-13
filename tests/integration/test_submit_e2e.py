import json
from pathlib import Path

from salvo import submit
from salvo.job.spec import JobSpec
from salvo.manifest.store import Manifest


def test_submit_pipeline_writes_artifact_dir(fake_sbatch, tmp_home):
    Manifest.load(Path(tmp_home, ".salvo", "manifest.toml"))  # load empty manifest
    spec = JobSpec(name="hello", cmd=["echo", "hi"], cpus=2, mem="4G", time="30m")
    h = submit(spec, cluster_id="mila")
    assert h.job_id.isdigit()
    assert h.artifact_dir.is_dir()
    assert (h.artifact_dir / "spec.json").exists()
    assert (h.artifact_dir / "cluster.json").exists()
    assert (h.artifact_dir / "sbatch.sh").exists()
    spec_dump = json.loads((h.artifact_dir / "spec.json").read_text())
    assert spec_dump["name"] == "hello"

    events = (h.artifact_dir / "events.jsonl").read_text().splitlines()
    names = [json.loads(e)["event"] for e in events]
    assert "submit.attempt" in names
    assert "submit.success" in names
    assert "dispatch.account_picked" in names
    assert "dispatch.partition_picked" in names
