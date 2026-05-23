"""Tests for `salvo.runtime.entrypoint`: payload parsing + target resolution + main()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from salvo.runtime import entrypoint

# ---------------------------------------------------------------------------
# parse_payload
# ---------------------------------------------------------------------------


def test_parse_payload_happy() -> None:
    raw = json.dumps({"target": "pkg.mod:fn", "kwargs": {"a": 1}})
    mod, attr, kwargs = entrypoint.parse_payload(raw)
    assert mod == "pkg.mod"
    assert attr == "fn"
    assert kwargs == {"a": 1}


def test_parse_payload_nested_attr() -> None:
    raw = json.dumps({"target": "pkg.mod:Outer.method", "kwargs": {}})
    mod, attr, kwargs = entrypoint.parse_payload(raw)
    assert mod == "pkg.mod"
    assert attr == "Outer.method"
    assert kwargs == {}


def test_parse_payload_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="invalid JSON payload"):
        entrypoint.parse_payload("not-json")


def test_parse_payload_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="object"):
        entrypoint.parse_payload(json.dumps([1, 2, 3]))


def test_parse_payload_missing_target() -> None:
    with pytest.raises(ValueError, match="'target'"):
        entrypoint.parse_payload(json.dumps({"kwargs": {}}))


def test_parse_payload_missing_kwargs() -> None:
    with pytest.raises(ValueError, match="'kwargs'"):
        entrypoint.parse_payload(json.dumps({"target": "foo:bar"}))


def test_parse_payload_target_wrong_type() -> None:
    with pytest.raises(ValueError, match=r"target.*str"):
        entrypoint.parse_payload(json.dumps({"target": 5, "kwargs": {}}))


def test_parse_payload_kwargs_wrong_type() -> None:
    with pytest.raises(ValueError, match=r"kwargs.*dict|dict.*kwargs"):
        entrypoint.parse_payload(json.dumps({"target": "foo:bar", "kwargs": "x"}))


def test_parse_payload_target_missing_colon() -> None:
    with pytest.raises(ValueError, match=r"module\.path:attribute"):
        entrypoint.parse_payload(json.dumps({"target": "foo", "kwargs": {}}))


def test_parse_payload_target_too_many_colons() -> None:
    with pytest.raises(ValueError, match=r"module\.path:attribute"):
        entrypoint.parse_payload(json.dumps({"target": "a:b:c", "kwargs": {}}))


def test_parse_payload_empty_target() -> None:
    with pytest.raises(ValueError, match=r"module\.path:attribute"):
        entrypoint.parse_payload(json.dumps({"target": "", "kwargs": {}}))


def test_parse_payload_empty_module() -> None:
    with pytest.raises(ValueError, match=r"module\.path:attribute"):
        entrypoint.parse_payload(json.dumps({"target": ":fn", "kwargs": {}}))


def test_parse_payload_empty_attr() -> None:
    with pytest.raises(ValueError, match=r"module\.path:attribute"):
        entrypoint.parse_payload(json.dumps({"target": "mod:", "kwargs": {}}))


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


def test_resolve_target_happy() -> None:
    fn = entrypoint.resolve_target("tests.unit._runtime_helpers", "simple")
    assert callable(fn)
    assert fn(a=1) == {"a": 1}


def test_resolve_target_nested() -> None:
    fn = entrypoint.resolve_target("tests.unit._runtime_helpers", "Nested.fn")
    assert callable(fn)
    assert fn(x=2) == {"x": 2}


def test_resolve_target_nonexistent_module() -> None:
    with pytest.raises(ImportError, match=r"nonexistent\.module"):
        entrypoint.resolve_target("nonexistent.module", "fn")


def test_resolve_target_nonexistent_attribute() -> None:
    with pytest.raises(AttributeError, match=r"nonexistent_attr.*json"):
        entrypoint.resolve_target("json", "nonexistent_attr")


def test_resolve_target_nonexistent_nested_attr() -> None:
    with pytest.raises(AttributeError, match="missing"):
        entrypoint.resolve_target("tests.unit._runtime_helpers", "Nested.missing")


def test_resolve_target_not_callable() -> None:
    with pytest.raises(TypeError, match="not callable"):
        entrypoint.resolve_target("tests.unit._runtime_helpers", "NOT_CALLABLE")


# ---------------------------------------------------------------------------
# main() — CLI contract
# ---------------------------------------------------------------------------


def test_main_no_argv(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([])
    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_main_bad_json(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(["not-json"])
    assert exc.value.code == 2
    assert "json" in capsys.readouterr().err.lower()


def test_main_missing_target(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"kwargs": {}})])
    assert exc.value.code == 2
    assert "target" in capsys.readouterr().err


def test_main_missing_kwargs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "foo:bar"})])
    assert exc.value.code == 2
    assert "kwargs" in capsys.readouterr().err


def test_main_target_wrong_type(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": 5, "kwargs": {}})])
    assert exc.value.code == 2
    assert "target" in capsys.readouterr().err


def test_main_kwargs_wrong_type(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "foo:bar", "kwargs": "x"})])
    assert exc.value.code == 2
    assert "kwargs" in capsys.readouterr().err


def test_main_target_missing_colon(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "foo", "kwargs": {}})])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "module.path:attribute" in err
    assert "'foo'" in err


def test_main_target_too_many_colons(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "a:b:c", "kwargs": {}})])
    assert exc.value.code == 2


def test_main_empty_target(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "", "kwargs": {}})])
    assert exc.value.code == 2


def test_main_nonexistent_module(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "nonexistent.module:fn", "kwargs": {}})])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "nonexistent.module" in err
    assert "cannot import" in err.lower()


def test_main_nonexistent_attribute(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "json:nonexistent_attr", "kwargs": {}})])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "nonexistent_attr" in err
    assert "json" in err


def test_main_not_callable(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        entrypoint.main([json.dumps({"target": "json:__name__", "kwargs": {}})])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not callable" in err
    assert "str" in err


def test_main_happy_path_calls_target(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    payload = json.dumps(
        {
            "target": "tests.unit._runtime_helpers:write_marker",
            "kwargs": {"path": str(marker)},
        }
    )
    # No SystemExit on success.
    entrypoint.main([payload])
    assert marker.read_text() == "ok"


def test_main_happy_nested_attr() -> None:
    payload = json.dumps({"target": "tests.unit._runtime_helpers:Nested.fn", "kwargs": {"y": 9}})
    entrypoint.main([payload])


def test_main_user_callable_exception_propagates() -> None:
    payload = json.dumps({"target": "tests.unit._runtime_helpers:raises_runtime", "kwargs": {}})
    with pytest.raises(RuntimeError, match="boom"):
        entrypoint.main([payload])


def test_main_uses_sys_argv_when_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker = tmp_path / "argv_marker.txt"
    payload = json.dumps(
        {
            "target": "tests.unit._runtime_helpers:write_marker",
            "kwargs": {"path": str(marker)},
        }
    )
    monkeypatch.setattr("sys.argv", ["salvo.runtime.entrypoint", payload])
    entrypoint.main()
    assert marker.read_text() == "ok"
