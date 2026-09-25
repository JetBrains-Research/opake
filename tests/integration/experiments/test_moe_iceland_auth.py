"""Offline cached-login checks with dummy credentials and the pinned SDK models."""

import io
import json
import logging
import os
import socket
import subprocess
import sys
import traceback
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
from deploy.zenml.moe_iceland import launch
from deploy.zenml.moe_iceland.settings import CLIENT_URL, STACK_ID

PRO_URL = "https://cloudapi.zenml.io"
WORKSPACE_TOKEN = "DummyToken-workspace"
PRO_TOKEN = "DummyToken-pro"
API_KEY = "DummyToken-api-key"
OTHER_TOKEN = "DummyToken-unrelated"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def deny(*args, **kwargs):
        raise AssertionError("Authentication tests must never contact a server.")

    monkeypatch.setattr(socket.socket, "connect", deny)
    for name in tuple(os.environ):
        if name.startswith("ZENML_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("ZENML_CONFIG_PATH", str(tmp_path / "parent-config"))
    monkeypatch.setenv("ZENML_ANALYTICS_OPT_IN", "false")
    monkeypatch.setenv("ZENML_ENABLE_REPO_INIT_WARNINGS", "false")
    monkeypatch.setenv("DISABLE_CREDENTIALS_DISK_CACHING", "true")


@pytest.fixture
def sdk(monkeypatch):
    zenml = pytest.importorskip("zenml")
    assert zenml.__version__ == "0.96.4"
    from zenml.login.credentials import APIToken, ServerCredentials
    from zenml.login.credentials_store import get_credentials_store

    store = get_credentials_store()
    monkeypatch.setattr(store, "credentials", {})
    monkeypatch.setattr(store, "last_modified_time", None)
    return SimpleNamespace(
        store=store, APIToken=APIToken, ServerCredentials=ServerCredentials
    )


@pytest.fixture
def request_plan():
    return {
        "profile": "gpu-probe-v1",
        "client_url": CLIENT_URL,
        "project_id": "357b8eb4-6c65-40d1-a4de-ea48a3279288",
        "authorization": {
            "deadline_utc": (datetime.now(UTC) + timedelta(days=3)).isoformat()
        },
    }


@pytest.fixture
def cached(sdk, request_plan):
    deadline = datetime.fromisoformat(request_plan["authorization"]["deadline_utc"])
    sdk.store.credentials = {
        CLIENT_URL: sdk.ServerCredentials(
            url=CLIENT_URL,
            api_token=sdk.APIToken(
                access_token=WORKSPACE_TOKEN,
                expires_in=3600,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
                leeway=180,
            ),
            organization_id=UUID("12345678-1234-5678-1234-567812345678"),
            workspace_id=UUID("87654321-4321-8765-4321-876543218765"),
            workspace_name="jcp-prod",
            pro_api_url=PRO_URL,
        ),
        PRO_URL: sdk.ServerCredentials(
            url=PRO_URL,
            pro_api_url=PRO_URL,
            api_token=sdk.APIToken(
                access_token=PRO_TOKEN,
                expires_in=30 * 86400,
                expires_at=deadline + timedelta(days=25),
                leeway=86400,
                device_id=UUID("11111111-2222-3333-4444-555555555555"),
                device_metadata={"name": "offline test"},
            ),
        ),
        "https://unrelated.invalid": sdk.ServerCredentials(
            url="https://unrelated.invalid", api_key=OTHER_TOKEN
        ),
    }
    return sdk.store


def test_cached_login_flag_is_explicit_and_gpu_only():
    assert not launch.parser().parse_args([]).cached_login
    args = launch.parser().parse_args(["--profile", "gpu-probe-v1", "--cached-login"])
    assert args.cached_login
    assert "cached_login" not in launch.make_plan(args)
    with pytest.raises(ValueError, match="GPU"):
        launch.make_plan(launch.parser().parse_args(["--cached-login"]))


def test_cached_round_trip_preserves_expiry_without_files(
    cached, request_plan, monkeypatch, tmp_path, capsys, caplog
):
    from deploy.zenml.moe_iceland import auth

    before = {url: record.model_dump() for url, record in cached.credentials.items()}
    environment = dict(os.environ)
    monkeypatch.delenv("DISABLE_CREDENTIALS_DISK_CACHING")
    # The real SDK getters must run before the parent disables its cache loader.
    getter = cached.get_credentials

    def load(url):
        assert "DISABLE_CREDENTIALS_DISK_CACHING" not in os.environ
        return getter(url)

    with monkeypatch.context() as patch:
        patch.setattr(cached, "get_credentials", load)
        payload = auth.collect_cached_auth(request_plan)
    assert isinstance(payload, bytes)
    assert PRO_TOKEN.encode() in payload
    assert OTHER_TOKEN.encode() not in payload
    assert {
        url: record.model_dump() for url, record in cached.credentials.items()
    } == before
    assert "DISABLE_CREDENTIALS_DISK_CACHING" not in os.environ

    auth.restore_cached_auth(io.BytesIO(payload), request_plan)
    assert os.environ["DISABLE_CREDENTIALS_DISK_CACHING"] == "true"
    assert set(cached.credentials) == {CLIENT_URL, PRO_URL}
    assert cached.can_login(CLIENT_URL)
    assert cached.get_api_key(CLIENT_URL) is None
    assert cached.get_pro_api_key(PRO_URL) is None
    for url in (CLIENT_URL, PRO_URL):
        assert cached.credentials[url].model_dump() == before[url]
    assert cached.last_modified_time is None
    # Exercise actual SDK persistence paths after restoration, not fake setters.
    cached.set_api_key(PRO_URL, API_KEY, is_zenml_pro=True)
    assert not list(tmp_path.rglob("credentials.yaml"))
    for file in tmp_path.rglob("*"):
        if file.is_file():
            assert b"DummyToken" not in file.read_bytes()
    output = capsys.readouterr()
    assert "DummyToken" not in output.out + output.err + caplog.text
    assert os.environ.get("HOME") == environment.get("HOME")


@pytest.mark.parametrize("expiry", ["missing", "naive", "expired", "short", "leeway"])
def test_pro_token_must_cover_authorization(cached, request_plan, expiry):
    from deploy.zenml.moe_iceland import auth

    token = cached.credentials[PRO_URL].api_token
    deadline = datetime.fromisoformat(request_plan["authorization"]["deadline_utc"])
    token.leeway = 0
    token.expires_at = {
        "missing": None,
        "naive": deadline.replace(tzinfo=None) + timedelta(days=1),
        "expired": datetime.now(UTC) - timedelta(seconds=1),
        "short": deadline - timedelta(seconds=1),
        "leeway": deadline + timedelta(seconds=60),
    }[expiry]
    if expiry == "leeway":
        token.leeway = 61
    with pytest.raises(ValueError, match="Cached ZenML login failed") as error:
        auth.collect_cached_auth(request_plan)
    assert "DummyToken" not in "".join(traceback.format_exception(error.value))


def test_effective_expiry_at_deadline_is_sufficient(cached, request_plan):
    from deploy.zenml.moe_iceland import auth

    token = cached.credentials[PRO_URL].api_token
    token.expires_at = datetime.fromisoformat(
        request_plan["authorization"]["deadline_utc"]
    ) + timedelta(seconds=token.leeway)
    payload = auth.collect_cached_auth(request_plan)
    auth.restore_cached_auth(io.BytesIO(payload), request_plan)
    assert cached.can_login(CLIENT_URL)


@pytest.mark.parametrize("url", [CLIENT_URL, PRO_URL])
def test_exact_context_api_keys_allow_refresh(cached, request_plan, url):
    from deploy.zenml.moe_iceland import auth

    cached.credentials[url].api_key = API_KEY
    cached.credentials[PRO_URL].api_token = None
    payload = auth.collect_cached_auth(request_plan)
    auth.restore_cached_auth(io.BytesIO(payload), request_plan)
    assert cached.can_login(CLIENT_URL)
    assert cached.get_api_key(url) == API_KEY


def test_workspace_token_and_unrelated_key_are_not_sufficient(cached, request_plan):
    from deploy.zenml.moe_iceland import auth

    del cached.credentials[PRO_URL]
    with pytest.raises(ValueError, match="Cached ZenML login failed"):
        auth.collect_cached_auth(request_plan)


@pytest.mark.parametrize(
    "variable",
    [
        "ZENML_STORE_API_KEY",
        "ZENML_STORE_API_TOKEN",
        "ZENML_STORE_USERNAME",
        "ZENML_STORE_PASSWORD",
        "ZENML_PRO_API_KEY",
        "ZENML_PRO_API_TOKEN",
    ],
)
def test_cached_login_rejects_competing_environment(
    request_plan, monkeypatch, tmp_path, variable
):
    monkeypatch.setenv(variable, API_KEY)
    before = dict(os.environ)
    with pytest.raises(ValueError, match="Do not combine cached login") as error:
        launch.submission_environment(
            request_plan, tmp_path / "config", tmp_path / "source", cached_login=True
        )
    assert API_KEY not in str(error.value)
    assert dict(os.environ) == before


def test_cached_child_environment_has_no_credentials(request_plan, tmp_path):
    before = dict(os.environ)
    env = launch.submission_environment(
        request_plan, tmp_path / "config", tmp_path / "source", cached_login=True
    )
    assert env["DISABLE_CREDENTIALS_DISK_CACHING"] == "true"
    assert env["ZENML_STORE_URL"] == CLIENT_URL
    assert env["ZENML_ACTIVE_STACK_ID"] == STACK_ID
    assert not any("DummyToken" in value for value in env.values())
    assert "ZENML_STORE_API_KEY" not in env
    assert "ZENML_STORE_API_TOKEN" not in env
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "change",
    [
        "third-server",
        "wrong-url",
        "wrong-pro-url",
        "unknown-field",
        "password",
        "expired",
    ],
)
def test_child_revalidates_payload(cached, request_plan, change):
    from deploy.zenml.moe_iceland import auth

    value = json.loads(auth.collect_cached_auth(request_plan))
    records = value["credentials"]
    if change == "third-server":
        records["https://unrelated.invalid"] = records[PRO_URL]
    elif change == "wrong-url":
        records[CLIENT_URL]["url"] = "https://unrelated.invalid"
    elif change == "wrong-pro-url":
        records[CLIENT_URL]["pro_api_url"] = "https://unrelated.invalid"
    elif change == "unknown-field":
        records[PRO_URL]["other_credential"] = OTHER_TOKEN
    elif change == "password":
        records[CLIENT_URL]["password"] = OTHER_TOKEN
    else:
        records[PRO_URL]["api_token"]["expires_at"] = datetime.now(UTC).isoformat()
    before = dict(cached.credentials)
    with pytest.raises(ValueError, match="Cached ZenML login failed") as error:
        auth.restore_cached_auth(io.BytesIO(json.dumps(value).encode()), request_plan)
    assert cached.credentials == before
    assert "DummyToken" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("payload", [b"", b"DummyToken-invalid-json", b"[]", b"\xff"])
def test_malformed_stdin_is_safe(request_plan, payload):
    from deploy.zenml.moe_iceland import auth

    with pytest.raises(ValueError, match="Cached ZenML login failed") as error:
        auth.restore_cached_auth(io.BytesIO(payload), request_plan)
    assert "DummyToken" not in "".join(traceback.format_exception(error.value))


def test_stdin_read_is_bounded(request_plan):
    from deploy.zenml.moe_iceland import auth

    stream = Mock()
    stream.read.return_value = b"x" * (auth.MAX_AUTH_BYTES + 1)
    with pytest.raises(ValueError, match="Cached ZenML login failed"):
        auth.restore_cached_auth(stream, request_plan)
    stream.read.assert_called_once_with(auth.MAX_AUTH_BYTES + 1)


def test_sdk_failure_does_not_leak_credentials(
    cached, request_plan, monkeypatch, capsys, caplog
):
    from deploy.zenml.moe_iceland import auth

    def fail(*args, **kwargs):
        logging.getLogger("zenml.login.credentials_store").error(PRO_TOKEN)
        raise RuntimeError(PRO_TOKEN)

    monkeypatch.setattr(cached, "get_credentials", fail)
    with pytest.raises(ValueError, match="Cached ZenML login failed") as error:
        auth.collect_cached_auth(request_plan)
    assert error.value.__suppress_context__
    output = capsys.readouterr()
    assert PRO_TOKEN not in output.out + output.err + caplog.text
    assert PRO_TOKEN not in "".join(traceback.format_exception(error.value))


def test_internal_stdin_flag_requires_staged_child(capsys):
    assert launch.main(["--_cached-auth-stdin"]) == 2
    assert "staged" in capsys.readouterr().err


def test_fresh_sdk_process_receives_only_stdin(cached, request_plan, tmp_path):
    from deploy.zenml.moe_iceland import auth

    payload = auth.collect_cached_auth(request_plan)
    env = launch.submission_environment(
        request_plan,
        tmp_path / "isolated-config",
        tmp_path / "source",
        cached_login=True,
    )
    env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    program = """
import json
import os
import socket
import sys

assert os.environ['DISABLE_CREDENTIALS_DISK_CACHING'] == 'true'
assert not any(name.startswith('zenml') for name in sys.modules)
def deny(*args, **kwargs):
    raise AssertionError('No network access allowed.')
socket.socket.connect = deny
from deploy.zenml.moe_iceland.auth import restore_cached_auth, PRO_API_URL
from deploy.zenml.moe_iceland.settings import CLIENT_URL
restore_cached_auth(sys.stdin.buffer, json.loads(sys.argv[1]))
from zenml.login.credentials_store import get_credentials_store
from zenml.models import OAuthTokenResponse
store = get_credentials_store()
assert store.can_login(CLIENT_URL)
assert set(store.credentials) == {CLIENT_URL, PRO_API_URL}
assert store.get_api_key(CLIENT_URL) is None
assert store.get_pro_api_key(PRO_API_URL) is None
token = store.get_pro_token(PRO_API_URL)
assert token.expires_at is not None and token.leeway is not None
store.set_token(PRO_API_URL, OAuthTokenResponse(
    access_token=token.access_token, token_type='bearer', expires_in=3600
), is_zenml_pro=True)
print('restored')
"""
    command = [sys.executable, "-c", program, json.dumps(request_plan)]
    result = subprocess.run(
        command, input=payload, env=env, capture_output=True, check=True, timeout=30
    )
    assert b"restored" in result.stdout
    assert b"DummyToken" not in result.stdout + result.stderr
    assert "DummyToken" not in json.dumps(command) + json.dumps(env)
    assert not list(tmp_path.rglob("credentials.yaml"))
    for file in tmp_path.rglob("*"):
        if file.is_file():
            assert b"DummyToken" not in file.read_bytes()


def test_launcher_handoff_keeps_credentials_out_of_artifacts(
    cached, request_plan, monkeypatch, tmp_path, capsys
):
    from deploy.zenml.moe_iceland import provenance

    root = tmp_path / "repo"
    scope = root / "deploy/zenml/moe_iceland"
    scope.mkdir(parents=True)
    monkeypatch.setattr(launch, "__file__", str(scope / "launch.py"))
    request_plan.update(
        run_name="offline-auth-test",
        source_sha256="c" * 64,
        timeout_seconds=900,
        accelerator="cuda",
        image="europe-west4-docker.pkg.dev/test-project/test-repo/moe@sha256:"
        + "a" * 64,
    )
    original = json.dumps(request_plan)
    monkeypatch.setattr(launch, "make_plan", Mock(return_value=request_plan))
    monkeypatch.setattr(launch, "authorize", Mock())
    monkeypatch.setattr(launch, "validate_config", Mock())
    monkeypatch.setattr(launch, "reserve_budget", Mock())
    monkeypatch.setattr(launch, "pipeline_settings", Mock(return_value={}))
    monkeypatch.setattr(provenance, "validate_contract", Mock())

    def stage(root, staged, **kwargs):
        staged.mkdir()
        return {"source_sha256": request_plan["source_sha256"]}

    monkeypatch.setattr(launch, "stage_source", stage)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "docker":
            assert "input" not in kwargs
            return SimpleNamespace(stdout='{"public_image_receipt":true}')
        assert command[-1] == "--_cached-auth-stdin"
        assert PRO_TOKEN.encode() in kwargs["input"]
        assert kwargs["env"]["DISABLE_CREDENTIALS_DISK_CACHING"] == "true"
        assert "DummyToken" not in json.dumps(command) + json.dumps(kwargs["env"])
        assert kwargs["check"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", run)
    assert launch.main(["--submit", "--profile", "gpu-probe-v1", "--cached-login"]) == 0
    assert len(calls) == 2
    plan_file = scope / "submissions/offline-auth-test/plan.json"
    assert json.loads(plan_file.read_text()) == {
        **json.loads(original),
        "image_contract": {"public_image_receipt": True},
    }
    for file in root.rglob("*"):
        if file.is_file():
            assert b"DummyToken" not in file.read_bytes()
    output = capsys.readouterr()
    assert "DummyToken" not in output.out + output.err


@pytest.mark.parametrize("fail", [False, True])
def test_staged_child_restores_before_submission_and_sanitizes_errors(
    cached, request_plan, monkeypatch, tmp_path, capsys, caplog, fail
):
    from deploy.zenml.moe_iceland import auth, provenance, remote, settings, source

    payload = auth.collect_cached_auth(request_plan)
    cached.credentials = {}
    staged = tmp_path / "source"
    plan_file = tmp_path / "plan.json"
    request_plan["image_contract"] = {}
    plan_file.write_text(json.dumps(request_plan))
    monkeypatch.setattr(
        launch, "__file__", str(staged / "deploy/zenml/moe_iceland/launch.py")
    )
    for name, value in launch.submission_environment(
        request_plan, tmp_path / "zenml-config", staged, cached_login=True
    ).items():
        if name != "HOME":
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        launch.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload))
    )
    monkeypatch.setattr(launch, "validate_execution_plan", Mock())
    monkeypatch.setattr(provenance, "validate_contract", Mock())
    monkeypatch.setattr(source, "validate_code_archive", Mock(return_value={}))

    def ready(*args):
        assert cached.can_login(CLIENT_URL)
        assert set(cached.credentials) == {CLIENT_URL, PRO_URL}
        assert os.environ["DISABLE_CREDENTIALS_DISK_CACHING"] == "true"

    monkeypatch.setattr(settings, "validate_native_settings", ready)

    def submit(*args):
        ready()
        logging.getLogger("zenml").error(PRO_TOKEN)
        if fail:
            raise RuntimeError(PRO_TOKEN)
        return {"status": "offline-test"}

    monkeypatch.setattr(remote, "submit_one", submit)
    assert launch.main(
        ["--submit", "--_submit-plan", str(plan_file), "--_cached-auth-stdin"]
    ) == (2 if fail else 0)
    output = capsys.readouterr()
    assert "DummyToken" not in output.out + output.err + caplog.text
    assert not list(tmp_path.rglob("credentials.yaml"))


def test_gpu_cached_dry_run_never_loads_credentials(monkeypatch, capsys):
    import builtins

    original = builtins.__import__

    def no_sdk(name, *args, **kwargs):
        if name == "zenml" or name.startswith("zenml."):
            raise AssertionError("Dry run must not import the SDK.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    collect = Mock(side_effect=AssertionError("Dry run must not read credentials."))
    monkeypatch.setattr(launch, "collect_cached_auth", collect)
    monkeypatch.setattr(launch, "inventory", Mock(return_value={}))
    assert (
        launch.main(["--dry-run", "--profile", "gpu-probe-v1", "--cached-login"]) == 0
    )
    collect.assert_not_called()
    assert json.loads(capsys.readouterr().out)["dry_run"]
