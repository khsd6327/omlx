# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin login, dashboard, and chat page handlers."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from omlx_web import routes as webui

import omlx.server  # noqa: F401 - wires the web UI host
import omlx.admin.routes as admin_routes


def _mock_global_settings(api_key=None):
    """Create a mock GlobalSettings with the given API key."""
    mock = MagicMock()
    mock.auth.api_key = api_key
    mock.auth.skip_api_key_verification = False
    mock.ui.language = "en"
    return mock


def _patch_getter(mock_settings):
    """Replace the module-level _get_global_settings with a lambda returning mock."""
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: mock_settings
    return original


def _restore_getter(original):
    """Restore the original _get_global_settings."""
    admin_routes._get_global_settings = original


class TestLoginPage:
    """Tests for GET /admin login page TemplateResponse signature."""

    def test_login_page_uses_new_template_signature(self):
        """login_page should pass request as first arg to TemplateResponse."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch("omlx.admin.routes.verify_session", return_value=False):
                with patch.object(webui, "templates") as mock_templates:
                    mock_templates.TemplateResponse.return_value = MagicMock()
                    asyncio.run(webui.login_page(request=mock_request))
                    mock_templates.TemplateResponse.assert_called_once_with(
                        mock_request, "login.html", {"api_key_configured": True}
                    )
        finally:
            _restore_getter(original)


class TestDashboardPage:
    """Tests for GET /admin/dashboard TemplateResponse signature."""

    def test_dashboard_page_uses_new_template_signature(self):
        """dashboard_page should pass request as first arg to TemplateResponse."""
        mock_request = MagicMock()
        with patch.object(webui, "templates") as mock_templates:
            mock_templates.TemplateResponse.return_value = MagicMock()
            asyncio.run(webui.dashboard_page(request=mock_request, is_admin=True))
            mock_templates.TemplateResponse.assert_called_once_with(
                mock_request, "dashboard.html", {"auth_source": "api_key"}
            )


class TestChatPageApiKeyInjection:
    """The chat page uses the session cookie; no API key reaches the template."""

    @pytest.mark.parametrize("api_key", ["test-chat-key", None])
    def test_chat_page_context_has_no_api_key(self, api_key):
        mock_settings = _mock_global_settings(api_key=api_key)
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch.object(webui, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(webui.chat_page(request=mock_request, is_admin=True))
                mock_templates.TemplateResponse.assert_called_once_with(
                    mock_request, "chat.html", {}
                )
        finally:
            _restore_getter(original)


class TestLoginPageSkipAuth:
    """Login page behavior when skip_api_key_verification is enabled."""

    def test_login_page_redirects_when_skip_enabled(self):
        """Login page should redirect to dashboard when skip is enabled on localhost."""
        gs = MagicMock()
        gs.auth.skip_api_key_verification = True
        gs.auth.api_key = "test-key"
        gs.ui.language = "en"
        gs.server.host = "127.0.0.1"
        original = _patch_getter(gs)
        try:
            mock_request = MagicMock()
            with patch("omlx.admin.routes.verify_session", return_value=False):
                result = asyncio.run(webui.login_page(request=mock_request))
                assert result.status_code == 302
                assert result.headers["location"] == "/admin/dashboard"
        finally:
            _restore_getter(original)

    def test_login_page_does_not_skip_login_on_network_host(self):
        gs = MagicMock()
        gs.auth.skip_api_key_verification = True
        gs.auth.api_key = "test-key"
        gs.ui.language = "en"
        gs.server.host = "0.0.0.0"
        original = _patch_getter(gs)
        rendered = MagicMock()
        try:
            mock_request = MagicMock()
            with (
                patch("omlx.admin.routes.verify_session", return_value=False),
                patch.object(
                    webui.templates,
                    "TemplateResponse",
                    return_value=rendered,
                ),
            ):
                result = asyncio.run(webui.login_page(request=mock_request))

            assert result is rendered
        finally:
            _restore_getter(original)
