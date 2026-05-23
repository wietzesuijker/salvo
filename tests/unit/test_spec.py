import pytest
from pydantic import ValidationError
from salvo.job.spec import JobSpec, PythonEntrypoint, parse_mem_mb, parse_time_min


@pytest.mark.parametrize(
    "s,mb",
    [
        ("4G", 4096),
        ("512M", 512),
        ("1.5G", 1536),
        ("32GB", 32768),
        ("16g", 16384),
        ("128MB", 128),
        ("4GiB", 4096),
        ("1TiB", 1048576),
        ("1T", 1048576),
        ("1.5T", 1572864),
        ("0.5G", 512),
    ],
)
def test_parse_mem(s, mb):
    assert parse_mem_mb(s) == mb


@pytest.mark.parametrize(
    "s,minutes",
    [
        ("1h", 60),
        ("90m", 90),
        ("2h30m", 150),
        ("3-00:00:00", 4320),
        ("12:00:00", 720),
        ("45s", 1),  # rounds up
        ("30", 30),
        ("5", 5),
        ("0h0m0s", 1),
    ],
)
def test_parse_time(s, minutes):
    assert parse_time_min(s) == minutes


def test_parse_time_zero_rejected():
    with pytest.raises(ValueError, match="unlimited"):
        parse_time_min("0")


def test_minimal_jobspec():
    s = JobSpec(name="t", cmd=["echo", "hi"])
    assert s.gpus == 0
    assert s.cpus == 1
    assert s.mem == "4G"
    assert s.time == "1h"
    assert s.on_preempt == "resubmit"
    assert s.max_hops == 5


def test_python_entrypoint():
    s = JobSpec(name="t", cmd=PythonEntrypoint(target="mymod:main", kwargs={"seed": 1}))
    assert isinstance(s.cmd, PythonEntrypoint)
    assert s.cmd.target == "mymod:main"


def test_python_entrypoint_target_format():
    with pytest.raises(ValidationError):
        PythonEntrypoint(target="not-a-dotted-path")  # missing colon


def test_jobspec_immutable():
    s = JobSpec(name="t", cmd=["echo"])
    with pytest.raises(ValidationError):
        s.name = "x"  # type: ignore[misc]


def test_mem_rejects_garbage():
    with pytest.raises(ValidationError):
        JobSpec(name="t", cmd=["echo"], mem="abc")


@pytest.mark.parametrize("bad", ["hi\nthere", "", "x\x00y", "a\rb", "tab\tname"])
def test_name_rejects_control_chars(bad):
    with pytest.raises(ValidationError):
        JobSpec(name=bad, cmd=["echo"])


@pytest.mark.parametrize("good", ["job1", "my-job", "my_job.v2", "abc"])
def test_name_accepts_normal(good):
    s = JobSpec(name=good, cmd=["echo"])
    assert s.name == good


@pytest.mark.parametrize("bad_key", ["FOO=bar", "FOO BAR", "FOO\nBAR", "1FOO", "", "foo-bar"])
def test_env_rejects_bad_keys(bad_key):
    with pytest.raises(ValidationError):
        JobSpec(name="t", cmd=["echo"], env={bad_key: "v"})


@pytest.mark.parametrize("good_key", ["FOO", "FOO_BAR", "_X", "a", "X1"])
def test_env_accepts_good_keys(good_key):
    s = JobSpec(name="t", cmd=["echo"], env={good_key: "v"})
    assert good_key in s.env
