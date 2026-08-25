# SPDX-License-Identifier: Apache-2.0
"""Static regressions preventing API-key material from returning to web clients."""

from pathlib import Path


ROOT = Path(__file__).parent.parent
CHAT = ROOT / "omlx/admin/templates/chat.html"
MENUBAR = ROOT / "apps/omlx-mac/Sources/Menubar/MenubarController.swift"


def test_chat_uses_session_cookie_instead_of_api_key():
    source = CHAT.read_text(encoding="utf-8")
    forbidden = (
        "apiKeySet",
        "apiKeyInput",
        "getApiKey()",
        "saveApiKey",
        "API_KEY_STORAGE_KEY",
        "{{ api_key",
        "'Authorization': `Bearer",
        "api_key: this.getApiKey",
    )
    for marker in forbidden:
        assert marker not in source
    assert "localStorage.removeItem('omlx_chat_api_key')" in source
    assert "credentials: 'same-origin'" in source


def test_open_dashboard_url_never_accepts_or_adds_api_key():
    source = MENUBAR.read_text(encoding="utf-8")
    function = source.split("static func webAdminURL", 1)[1].split(
        "static func shouldShowGenericFailureAlert", 1
    )[0]
    assert "apiKey" not in function
    assert 'URLQueryItem(name: "key"' not in function
