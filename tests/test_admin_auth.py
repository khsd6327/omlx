# SPDX-License-Identifier: Apache-2.0
"""Tests for admin authentication and trusted-network web sessions."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import omlx.server  # noqa: F401 — ensure server module is imported first
import omlx.admin.auth as admin_auth
import omlx.admin.routes as admin_routes


def _mock_global_settings(api_key=None):
    """Create a mock GlobalSettings with the given API key."""
    mock = MagicMock()
    mock.auth.api_key = api_key
    mock.auth.skip_api_key_verification = False
    mock.auth.web_admin_auth_mode = "api_key"
    return mock


def _request(client="127.0.0.1", headers=None, scheme="http"):
    raw_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/admin",
            "scheme": scheme,
            "server": ("localhost", 8000),
            "client": (client, 12345),
            "headers": raw_headers,
        }
    )


def _patch_getter(mock_settings):
    """Replace the module-level _get_global_settings with a lambda returning mock."""
    original = (
        admin_routes._get_global_settings,
        admin_auth._get_global_settings,
    )
    admin_routes._get_global_settings = lambda: mock_settings
    admin_auth._get_global_settings = lambda: mock_settings
    return original


def _restore_getter(original):
    """Restore the original _get_global_settings."""
    admin_routes._get_global_settings, admin_auth._get_global_settings = original


class TestAutoLogin:
    """Tests for GET /admin/auto-login endpoint."""

    def test_trusted_peer_redirects_to_dashboard(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        mock_settings.auth.web_admin_auth_mode = "trusted_networks"
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(
                    request=_request(), redirect="/admin/dashboard"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/dashboard"
            # Check that session cookie is set
            cookie_header = result.headers.get("set-cookie", "")
            assert "omlx_admin_session" in cookie_header
            assert "HttpOnly" in cookie_header
            assert "SameSite=strict" in cookie_header
            assert "Path=/" in cookie_header
        finally:
            _restore_getter(original)

    def test_trusted_peer_redirects_to_chat(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        mock_settings.auth.web_admin_auth_mode = "trusted_networks"
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(request=_request("100.100.101.1"), redirect="/admin/chat")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/chat"
        finally:
            _restore_getter(original)

    def test_public_peer_redirects_to_login_even_with_legacy_key_query(self):
        mock_settings = _mock_global_settings(api_key="correct-key")
        mock_settings.auth.web_admin_auth_mode = "trusted_networks"
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(
                    request=_request("8.8.8.8"), redirect="/admin/dashboard"
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
            cookie_header = result.headers.get("set-cookie", "")
            assert "omlx_admin_session" not in cookie_header
        finally:
            _restore_getter(original)

    def test_api_key_mode_redirects_to_login(self):
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(request=_request(), redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)

    def test_forwarded_header_disables_trusted_bootstrap(self):
        mock_settings = _mock_global_settings(api_key=None)
        mock_settings.auth.web_admin_auth_mode = "trusted_networks"
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(
                    request=_request(headers={"X-Forwarded-For": "127.0.0.1"}),
                    redirect="/admin/dashboard",
                )
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)

    def test_auto_login_invalid_redirect_returns_400(self):
        """Redirect path not starting with /admin should return 400."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.auto_login(
                        request=_request(), redirect="https://evil.com"
                    )
                )
            assert exc_info.value.status_code == 400
            assert "Invalid redirect path" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_auto_login_redirect_to_admin_root(self):
        """Redirect to /admin (exact match) should be allowed."""
        mock_settings = _mock_global_settings(api_key="test-key")
        mock_settings.auth.web_admin_auth_mode = "trusted_networks"
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(request=_request(), redirect="/admin")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)


class TestLoginPage:
    """Tests for GET /admin login page TemplateResponse signature."""

    def test_login_page_uses_new_template_signature(self):
        """login_page should pass request as first arg to TemplateResponse."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            mock_request = _request()
            with patch("omlx.admin.auth.verify_session", return_value=False):
                with patch.object(admin_routes, "templates") as mock_templates:
                    mock_templates.TemplateResponse.return_value = MagicMock()
                    asyncio.run(admin_routes.login_page(request=mock_request))
                    mock_templates.TemplateResponse.assert_called_once_with(
                        mock_request, "login.html", {"api_key_configured": True}
                    )
        finally:
            _restore_getter(original)


class TestDashboardPage:
    """Tests for GET /admin/dashboard TemplateResponse signature."""

    def test_dashboard_page_uses_new_template_signature(self):
        """dashboard_page should pass request as first arg to TemplateResponse."""
        mock_request = _request()
        with patch.object(admin_routes, "templates") as mock_templates:
            mock_templates.TemplateResponse.return_value = MagicMock()
            asyncio.run(
                admin_routes.dashboard_page(request=mock_request, is_admin=True)
            )
            mock_templates.TemplateResponse.assert_called_once_with(
                mock_request, "dashboard.html", {"auth_source": "api_key"}
            )


class TestChatPageAuthentication:
    """Tests for GET /admin/chat without API key template injection."""

    def test_chat_page_does_not_pass_api_key_in_context(self):
        mock_settings = _mock_global_settings(api_key="test-chat-key")
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                mock_templates.TemplateResponse.assert_called_once_with(
                    mock_request,
                    "chat.html",
                    {},
                )
        finally:
            _restore_getter(original)

    def test_chat_page_context_stays_empty_without_key(self):
        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                assert mock_templates.TemplateResponse.call_args[0][2] == {}
        finally:
            _restore_getter(original)

class TestSkipAdminAuth:
    """Tests for skipping admin auth when skip_api_key_verification is enabled."""

    def _mock_gs(self, skip=True, host="127.0.0.1"):
        mock = MagicMock()
        mock.auth.skip_api_key_verification = skip
        mock.server.host = host
        return mock

    def test_require_admin_skipped_on_localhost(self):
        """require_admin should pass when skip_api_key_verification=True."""
        gs = self._mock_gs(skip=True, host="127.0.0.1")
        original = admin_auth._get_global_settings
        admin_auth._get_global_settings = lambda: gs
        try:
            mock_request = MagicMock()
            mock_request.cookies.get.return_value = None  # No session cookie
            result = asyncio.run(admin_auth.require_admin(mock_request))
            assert result is True
        finally:
            admin_auth._get_global_settings = original

    def test_require_admin_skipped_on_any_host(self):
        """require_admin should skip auth when skip_api_key_verification=True regardless of host."""
        gs = self._mock_gs(skip=True, host="0.0.0.0")
        original = admin_auth._get_global_settings
        admin_auth._get_global_settings = lambda: gs
        try:
            mock_request = MagicMock()
            mock_request.cookies.get.return_value = None
            result = asyncio.run(admin_auth.require_admin(mock_request))
            assert result is True
        finally:
            admin_auth._get_global_settings = original

    def test_require_admin_not_skipped_when_disabled(self):
        """require_admin should still require auth when skip_api_key_verification=False."""
        gs = self._mock_gs(skip=False, host="127.0.0.1")
        original = admin_auth._get_global_settings
        admin_auth._get_global_settings = lambda: gs
        try:
            mock_request = MagicMock()
            mock_request.cookies.get.return_value = None
            mock_request.headers.get.return_value = "application/json"
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_auth.require_admin(mock_request))
            assert exc_info.value.status_code == 401
        finally:
            admin_auth._get_global_settings = original

    def test_login_page_redirects_when_skip_enabled(self):
        """Login page should redirect to dashboard when skip is enabled on localhost."""
        gs = MagicMock()
        gs.auth.skip_api_key_verification = True
        gs.auth.api_key = "test-key"
        gs.server.host = "127.0.0.1"
        original = _patch_getter(gs)
        try:
            mock_request = MagicMock()
            with patch("omlx.admin.auth.verify_session", return_value=False):
                result = asyncio.run(admin_routes.login_page(request=mock_request))
                assert result.status_code == 302
                assert result.headers["location"] == "/admin/dashboard"
        finally:
            _restore_getter(original)


class TestInitAuth:
    """Tests for init_auth() persistent secret key initialization."""

    def test_init_auth_sets_serializer(self):
        """init_auth should update the serializer with the provided key."""
        original_serializer = admin_auth._serializer
        try:
            admin_auth.init_auth("test-persistent-secret-key")
            # Create a token with the new serializer
            token = admin_auth.create_session_token()
            assert admin_auth.verify_session_token(token) is True
        finally:
            admin_auth._serializer = original_serializer

    def test_init_auth_env_var_takes_priority(self):
        """OMLX_SECRET_KEY env var should take priority over provided key."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            with patch.dict("os.environ", {"OMLX_SECRET_KEY": "env-secret-key"}):
                admin_auth.init_auth("settings-secret-key")
                assert admin_auth.SECRET_KEY == "env-secret-key"
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret

    def test_init_auth_uses_provided_key_when_no_env(self):
        """Should use provided key when no OMLX_SECRET_KEY env var."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            with patch.dict("os.environ", {}, clear=True):
                # Remove OMLX_SECRET_KEY if it exists
                import os

                os.environ.pop("OMLX_SECRET_KEY", None)
                admin_auth.init_auth("my-persistent-key")
                assert admin_auth.SECRET_KEY == "my-persistent-key"
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret

    def test_tokens_survive_reinit_with_same_key(self):
        """Tokens created before re-init should still be valid with same key."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            key = "persistent-key-for-test"
            admin_auth.init_auth(key)
            token = admin_auth.create_session_token()

            # Re-initialize with same key (simulates server restart)
            admin_auth.init_auth(key)
            assert admin_auth.verify_session_token(token) is True
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret

    def test_tokens_invalid_after_reinit_with_different_key(self):
        """Tokens should be invalid after re-init with a different key."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            admin_auth.init_auth("key-one")
            token = admin_auth.create_session_token()

            admin_auth.init_auth("key-two")
            assert admin_auth.verify_session_token(token) is False
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret


class TestRememberMe:
    """Tests for remember me session token functionality."""

    def test_create_token_default_no_remember(self):
        """Default token should not have remember flag."""
        token = admin_auth.create_session_token()
        # Verify it works with default max_age
        assert admin_auth.verify_session_token(token) is True

    def test_create_token_with_remember(self):
        """Token with remember=True should be valid."""
        token = admin_auth.create_session_token(remember=True)
        assert admin_auth.verify_session_token(token) is True

    def test_remember_token_has_extended_max_age(self):
        """Remember token should use 30-day max_age for verification."""
        token = admin_auth.create_session_token(remember=True)
        # Manually load the payload to check the remember flag
        data = admin_auth._serializer.loads(token, max_age=None)
        assert data["remember"] is True
        assert data["admin"] is True

    def test_non_remember_token_payload(self):
        """Non-remember token should have remember=False in payload."""
        token = admin_auth.create_session_token(remember=False)
        data = admin_auth._serializer.loads(token, max_age=None)
        assert data["remember"] is False
        assert data["admin"] is True

    def test_remember_me_max_age_constant(self):
        """REMEMBER_ME_MAX_AGE should be 30 days."""
        assert admin_auth.REMEMBER_ME_MAX_AGE == 2592000  # 30 * 24 * 60 * 60

    def test_session_max_age_constant(self):
        """SESSION_MAX_AGE should be 24 hours."""
        assert admin_auth.SESSION_MAX_AGE == 86400  # 24 * 60 * 60


# =============================================================================
# Update Check
# =============================================================================


def _make_async_return(value):
    """Create a coroutine function that returns the given value."""

    async def _coro(*args, **kwargs):
        return value

    return _coro


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


class TestCheckUpdate:
    """Tests for update-check version filtering."""

    def setup_method(self):
        admin_routes._update_cache = {}
        admin_routes._update_cache_time = {}
        admin_routes._UPDATE_PREFS_PATH = Path(
            "/tmp/omlx-test-missing-update-prefs.json"
        )

    @pytest.mark.asyncio
    async def test_prerelease_not_shown(self):
        """Dev/pre-release GitHub releases should not trigger update notification."""
        fake_resp = _FakeResponse(
            200,
            [{
                "tag_name": "v99.0.0.dev1",
                "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0.dev1",
            }],
        )
        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None

    @pytest.mark.asyncio
    async def test_stable_version_shown(self):
        """Stable GitHub releases should trigger update notification."""
        fake_resp = _FakeResponse(
            200,
            [{
                "tag_name": "v99.0.0",
                "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0",
            }],
        )
        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is True
        assert result["latest_version"] == "99.0.0"

    @pytest.mark.asyncio
    async def test_rc_not_shown(self):
        """RC releases should not trigger update notification."""
        fake_resp = _FakeResponse(
            200,
            [{
                "tag_name": "v99.0.0rc1",
                "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0rc1",
            }],
        )
        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None
