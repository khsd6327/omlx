# SPDX-License-Identifier: Apache-2.0
"""Unit tests for trusted-network browser authentication."""

from types import SimpleNamespace

import pytest
from starlette.requests import Request

from omlx.admin import auth


def _request(
    client: str,
    *,
    headers: dict[str, str] | None = None,
    scheme: str = "http",
) -> Request:
    values = {"host": "omlx.local:8000", **(headers or {})}
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "scheme": scheme,
            "server": ("omlx.local", 8000),
            "client": (client, 12345),
            "headers": [
                (key.lower().encode(), value.encode())
                for key, value in values.items()
            ],
        }
    )


@pytest.fixture(autouse=True)
def trusted_mode():
    original = auth._get_global_settings
    settings = SimpleNamespace(
        auth=SimpleNamespace(
            web_admin_auth_mode="trusted_networks",
            skip_api_key_verification=False,
        )
    )
    auth._get_global_settings = lambda: settings
    try:
        yield settings
    finally:
        auth._get_global_settings = original


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.42.0.1",
        "::1",
        "10.1.2.3",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.50.2",
        "100.64.0.1",
        "100.127.255.254",
        "fd7a:115c:a1e0::1",
        "fe80::1%en0",
    ],
)
def test_trusted_network_ranges(address):
    assert auth.is_trusted_web_client(_request(address)) is True


@pytest.mark.parametrize(
    "address",
    ["8.8.8.8", "172.15.255.255", "172.32.0.1", "100.128.0.1", "2001:4860::1"],
)
def test_public_addresses_are_not_trusted(address):
    assert auth.is_trusted_web_client(_request(address)) is False


@pytest.mark.parametrize("header", auth.FORWARDED_HEADERS)
def test_forwarding_headers_disable_trusted_auth(header):
    request = _request("127.0.0.1", headers={header: "127.0.0.1"})
    assert auth.is_trusted_web_client(request) is False


def test_invalid_peer_address_is_not_trusted():
    assert auth.is_trusted_web_client(_request("not-an-ip")) is False


def test_api_key_mode_disables_trusted_auth(trusted_mode):
    trusted_mode.auth.web_admin_auth_mode = "api_key"
    assert auth.is_trusted_web_client(_request("127.0.0.1")) is False


def test_trusted_session_is_rechecked_on_every_request():
    token = auth.create_session_token(auth_source="trusted_network")
    headers = {"cookie": f"{auth.SESSION_COOKIE_NAME}={token}"}
    assert auth.verify_session(_request("192.168.1.2", headers=headers)) is True
    assert auth.verify_session(_request("8.8.8.8", headers=headers)) is False


def test_legacy_session_is_treated_as_api_key_session():
    token = auth._serializer.dumps({"admin": True, "remember": False})
    request = _request(
        "8.8.8.8", headers={"cookie": f"{auth.SESSION_COOKIE_NAME}={token}"}
    )
    assert auth.verify_session(request) is True
    assert auth.session_auth_source(request) == "api_key"


def test_same_origin_session_is_accepted_for_api():
    token = auth.create_session_token(auth_source="trusted_network")
    request = _request(
        "100.100.101.1",
        headers={
            "cookie": f"{auth.SESSION_COOKIE_NAME}={token}",
            "origin": "http://omlx.local:8000",
            "sec-fetch-site": "same-origin",
        },
    )
    assert auth.verify_same_origin_session(request) is True


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://evil.example", "sec-fetch-site": "cross-site"},
        {"origin": "http://omlx.local:8000", "sec-fetch-site": "cross-site"},
        {},
    ],
)
def test_cross_origin_or_non_browser_session_is_rejected(headers):
    token = auth.create_session_token(auth_source="trusted_network")
    request = _request(
        "100.100.101.1",
        headers={"cookie": f"{auth.SESSION_COOKIE_NAME}={token}", **headers},
    )
    assert auth.verify_same_origin_session(request) is False
