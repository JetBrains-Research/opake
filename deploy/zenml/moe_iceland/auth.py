"""Bounded, memory-only transfer of an explicitly selected cached ZenML login."""

import json
import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import BinaryIO

from .errors import refuse
from .gpu_policy import GPU_PROFILES
from .settings import CLIENT_URL

PRO_API_URL = "https://cloudapi.zenml.io"
DISABLE_CACHE = "DISABLE_CREDENTIALS_DISK_CACHING"
MAX_AUTH_BYTES = 64 * 1024


@contextmanager
def protected_auth():
    """Prevent SDK validation/network errors and credential-cache logs escaping."""
    previous = logging.root.manager.disable
    logging.disable(max(previous, logging.CRITICAL))
    try:
        yield
    except Exception:
        message = (
            "Cached ZenML login failed; verify the cached login and authorization "
            "deadline, and inspect public run state before retrying."
        )
        raise ValueError(message) from None
    finally:
        logging.disable(previous)


def validate_cached_login(plan: dict, environ: dict) -> None:
    """Accept only the confirmed GPU target without competing authentication."""
    if plan.get("profile") not in GPU_PROFILES:
        refuse("Cached login is restricted to GPU profiles.")
    if (
        plan.get("client_url") != CLIENT_URL
        or environ.get("ZENML_STORE_URL", CLIENT_URL).rstrip("/") != CLIENT_URL
    ):
        refuse("Cached login requires the confirmed laptop ZenML URL.")
    if environ.get("ZENML_PRO_API_URL", PRO_API_URL) != PRO_API_URL:
        refuse("Cached login requires the confirmed ZenML Pro API URL.")
    if any(
        value
        and name.startswith("ZENML_")
        and name.endswith(("_TOKEN", "_KEY", "_USERNAME", "_PASSWORD"))
        for name, value in environ.items()
    ):
        refuse(
            "Do not combine cached login with ZenML authentication environment values."
        )
    try:
        deadline = datetime.fromisoformat(plan["authorization"]["deadline_utc"])
        if deadline.tzinfo is None or deadline <= datetime.now(UTC):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        message = "Cached login requires a future timezone-qualified deadline."
        raise ValueError(message) from None


def _validated_records(value: dict, plan: dict) -> dict:
    import zenml
    from zenml.constants import ENV_ZENML_DISABLE_CREDENTIALS_DISK_CACHING
    from zenml.login.credentials import APIToken, ServerCredentials, ServerType
    from zenml.login.pro.constants import ZENML_PRO_API_URL

    if (
        zenml.__version__ != "0.96.4"
        or ZENML_PRO_API_URL != PRO_API_URL
        or ENV_ZENML_DISABLE_CREDENTIALS_DISK_CACHING != DISABLE_CACHE
    ):
        refuse("Cached login requires the pinned ZenML SDK and Pro endpoint.")
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "credentials"}
        or type(value["schema"]) is not int
        or value["schema"] != 1
    ):
        refuse("Invalid cached login payload.")
    raw = value["credentials"]
    if (
        not isinstance(raw, dict)
        or CLIENT_URL not in raw
        or not set(raw) <= {CLIENT_URL, PRO_API_URL}
    ):
        refuse("Invalid cached login contexts.")
    records = {}
    for url, data in raw.items():
        if (
            not isinstance(data, dict)
            or not set(data) <= ServerCredentials.model_fields.keys()
        ):
            refuse("Invalid cached server credentials.")
        token = data.get("api_token")
        if token is not None and (
            not isinstance(token, dict)
            or not set(token) <= APIToken.model_fields.keys()
        ):
            refuse("Invalid cached token metadata.")
        record = ServerCredentials.model_validate(data)
        if (
            record.url != url
            or record.pro_api_url not in (None, PRO_API_URL)
            or record.username is not None
            or record.password is not None
            or (record.api_key is not None and not record.api_key.strip())
        ):
            refuse("Unconfirmed cached authentication context.")
        if record.api_token is not None and (
            not record.api_token.access_token.strip()
            or (record.api_token.leeway is not None and record.api_token.leeway < 0)
        ):
            refuse("Invalid cached token.")
        records[url] = record
    workspace = records[CLIENT_URL]
    pro = records.get(PRO_API_URL)
    if pro is not None and pro.type != ServerType.PRO_API:
        refuse("Unconfirmed Pro credential type.")
    if workspace.api_key:
        return records
    if (
        workspace.type != ServerType.PRO
        or workspace.pro_api_url != PRO_API_URL
        or pro is None
    ):
        refuse("Cached workspace login cannot be refreshed.")
    if not pro.api_key:
        token = pro.api_token
        expiry = token.expires_at_with_leeway if token else None
        deadline = datetime.fromisoformat(plan["authorization"]["deadline_utc"])
        if expiry is None or expiry.tzinfo is None or expiry < deadline:
            refuse("Cached Pro login does not cover the authorization deadline.")
    return records


def collect_cached_auth(plan: dict) -> bytes:
    """Read only exact SDK cache contexts, before disabling the parent's loader."""
    with protected_auth():
        validate_cached_login(plan, os.environ)
        from zenml.login.credentials import APIToken
        from zenml.login.credentials_store import get_credentials_store

        store = get_credentials_store()
        records = {}
        for url in (CLIENT_URL, PRO_API_URL):
            record = store.get_credentials(url)
            if record is None:
                continue
            data = record.model_dump(mode="json", exclude={"username", "password"})
            data["api_key"] = (
                store.get_pro_api_key(url)
                if url == PRO_API_URL
                else store.get_api_key(url)
            )
            token = (
                store.get_pro_token(url) if url == PRO_API_URL else store.get_token(url)
            )
            data["api_token"] = (
                token.model_dump(mode="json", include=set(APIToken.model_fields))
                if token is not None
                else None
            )
            records[url] = data
        value = {"schema": 1, "credentials": records}
        _validated_records(value, plan)
        payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        if len(payload) > MAX_AUTH_BYTES:
            refuse("Oversized cached login payload.")
        return payload


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            refuse("Duplicate cached login field.")
        result[key] = value
    return result


def restore_cached_auth(stream: BinaryIO, plan: dict) -> None:
    """Restore only validated records in the isolated child, never on disk."""
    os.environ[DISABLE_CACHE] = "true"
    with protected_auth():
        validate_cached_login(plan, os.environ)
        payload = stream.read(MAX_AUTH_BYTES + 1)
        if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_AUTH_BYTES:
            refuse("Invalid cached login payload size.")
        records = _validated_records(
            json.loads(payload, object_pairs_hook=_unique_object), plan
        )
        from zenml.login.credentials_store import get_credentials_store

        store = get_credentials_store()
        store.credentials = records
        store.last_modified_time = None
        if not store.can_login(CLIENT_URL):
            store.credentials = {}
            refuse("Restored cached login cannot refresh the workspace session.")
