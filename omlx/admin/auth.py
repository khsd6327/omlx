# SPDX-License-Identifier: Apache-2.0
"""Authentication utilities for the oMLX admin panel.

This module provides session-based authentication using signed tokens
and API key verification for admin panel access.
"""

import hashlib
import ipaddress
import os
import secrets
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

# Session configuration
SESSION_COOKIE_NAME = "omlx_admin_session"
SESSION_MAX_AGE = 86400  # 24 hours in seconds
REMEMBER_ME_MAX_AGE = 2592000  # 30 days in seconds

# Secret key for signing session tokens
# Use environment variable if set, otherwise generate a random key
# Note: Random key means sessions won't persist across server restarts
# This is a fallback; init_auth() should be called with a persistent key
SECRET_KEY = os.environ.get("OMLX_SECRET_KEY") or secrets.token_hex(32)

# Initialize the serializer for creating and verifying session tokens
_serializer = URLSafeTimedSerializer(SECRET_KEY)

# Global settings getter (set by init_auth)
_get_global_settings = None

TRUSTED_WEB_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "127.0.0.0/8",
        "::1/128",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "fc00::/7",
        "fe80::/10",
    )
)
FORWARDED_HEADERS = ("forwarded", "x-forwarded-for", "x-real-ip")


def init_auth(secret_key: str, global_settings_getter=None) -> None:
    """Initialize authentication with a persistent secret key.

    Should be called during server startup with the secret key from settings.
    Environment variable OMLX_SECRET_KEY takes priority if set.

    Args:
        secret_key: The secret key from settings.json for signing tokens.
        global_settings_getter: Optional callable that returns GlobalSettings.
    """
    global _serializer, SECRET_KEY, _get_global_settings
    # Environment variable takes priority over settings
    key = os.environ.get("OMLX_SECRET_KEY") or secret_key
    SECRET_KEY = key
    _serializer = URLSafeTimedSerializer(key)
    if global_settings_getter is not None:
        _get_global_settings = global_settings_getter


def create_session_token(
    remember: bool = False, auth_source: str = "api_key"
) -> str:
    """Create a signed session token for admin authentication.

    Args:
        remember: If True, the token payload includes a remember flag
                  for extended session duration (30 days).

    Returns:
        A URL-safe signed token string containing admin session data.

    Example:
        >>> token = create_session_token()
        >>> verify_session_token(token)
        True
    """
    if auth_source not in {"api_key", "trusted_network"}:
        raise ValueError("Invalid admin session auth source")
    payload = {
        "admin": True,
        "remember": remember,
        "auth_source": auth_source,
    }
    return _serializer.dumps(payload)


def decode_session_token(
    token: str, max_age: int = SESSION_MAX_AGE
) -> dict[str, Any] | None:
    """Decode a valid session token, treating legacy tokens as API-key sessions."""
    try:
        data = _serializer.loads(token, max_age=None)
        if data.get("admin", False) is not True:
            return None
        effective_max_age = (
            REMEMBER_ME_MAX_AGE if data.get("remember", False) else max_age
        )
        data = _serializer.loads(token, max_age=effective_max_age)
        if data.get("admin", False) is not True:
            return None
        data.setdefault("auth_source", "api_key")
        if data["auth_source"] not in {"api_key", "trusted_network"}:
            return None
        return data
    except (BadSignature, SignatureExpired):
        return None


def verify_session_token(token: str, max_age: int = SESSION_MAX_AGE) -> bool:
    """Verify and decode a session token.

    The max_age is determined by the token's remember flag:
    - remember=True: 30 days
    - remember=False (default): 24 hours

    Args:
        token: The signed session token to verify.
        max_age: Maximum age of the token in seconds. Defaults to 24 hours.
                 This is overridden by the token's remember flag.

    Returns:
        True if the token is valid and not expired, False otherwise.

    Example:
        >>> token = create_session_token()
        >>> verify_session_token(token)
        True
        >>> verify_session_token("invalid_token")
        False
    """
    return decode_session_token(token, max_age=max_age) is not None


def trusted_web_auth_enabled() -> bool:
    if _get_global_settings is None:
        return False
    settings = _get_global_settings()
    return bool(
        settings is not None
        and settings.auth.web_admin_auth_mode == "trusted_networks"
    )


def is_trusted_web_client(connection: Any) -> bool:
    """Check the actual TCP peer against the explicitly trusted networks."""
    if not trusted_web_auth_enabled():
        return False
    if any(connection.headers.get(name) is not None for name in FORWARDED_HEADERS):
        return False
    client = getattr(connection, "client", None)
    host = getattr(client, "host", None)
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return any(address in network for network in TRUSTED_WEB_NETWORKS)


def _origin_matches_connection(connection: Any) -> bool:
    origin = connection.headers.get("origin")
    if not origin:
        return False
    parsed = urlsplit(origin)
    host = connection.headers.get("host")
    if not parsed.scheme or not parsed.netloc or not host:
        return False
    connection_scheme = getattr(getattr(connection, "url", None), "scheme", "")
    if connection_scheme == "ws":
        connection_scheme = "http"
    elif connection_scheme == "wss":
        connection_scheme = "https"
    return parsed.scheme == connection_scheme and parsed.netloc == host


def verify_same_origin_session(request: Request) -> bool:
    """Allow an admin cookie on browser API calls from the same origin only."""
    if not verify_session(request):
        return False
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site != "same-origin":
        return False
    origin = request.headers.get("origin")
    if origin is not None and not _origin_matches_connection(request):
        return False
    return fetch_site == "same-origin" or origin is not None


def verify_websocket_session(websocket: Any) -> bool:
    """Validate a browser WebSocket using its admin cookie and Origin."""
    return verify_session(websocket) and _origin_matches_connection(websocket)


def compare_keys(provided_key: str, expected_key: str) -> bool:
    """Compare two API keys in constant time, tolerating any str input.

    secrets.compare_digest raises TypeError when given str arguments that
    contain non-ASCII characters, which turns a bad client key into an
    unhandled 500 instead of a 401. Comparing UTF-8 bytes accepts any
    input while keeping the constant-time guarantee. surrogatepass covers
    lone surrogates, which json.loads can produce from escape sequences
    and which strict UTF-8 encoding rejects.

    Both arguments must be str; None is the caller's responsibility.

    Args:
        provided_key: The key supplied by the client (untrusted).
        expected_key: The configured key to compare against.

    Returns:
        True if the keys match, False otherwise.
    """
    return secrets.compare_digest(
        provided_key.encode("utf-8", "surrogatepass"),
        expected_key.encode("utf-8", "surrogatepass"),
    )


def fingerprint_key(api_key: str) -> str:
    """Return a short, non-reversible fingerprint of an API key for logging.

    Logging a rejected key verbatim leaks the client's secret into the server
    log. A truncated SHA-256 digest lets operators correlate repeated
    rejections of the same key without exposing the key itself. surrogatepass
    matches compare_keys() so any str the auth path accepts can be
    fingerprinted, including lone surrogates from json escape sequences.

    Args:
        api_key: The (untrusted) key to fingerprint. Empty string is allowed.

    Returns:
        The first 8 hex characters of the SHA-256 digest of the UTF-8 bytes.
    """
    digest = hashlib.sha256(api_key.encode("utf-8", "surrogatepass")).hexdigest()
    return digest[:8]


def verify_api_key(api_key: str, server_api_key: str) -> bool:
    """Verify an API key using constant-time comparison.

    This function uses constant-time comparison to prevent timing attacks
    when comparing the provided API key with the server's API key.

    Args:
        api_key: The API key provided by the client.
        server_api_key: The server's configured API key.

    Returns:
        True if the API keys match, False otherwise.

    Example:
        >>> verify_api_key("secret123", "secret123")
        True
        >>> verify_api_key("wrong", "secret123")
        False
    """
    if not api_key or not server_api_key:
        return False
    return compare_keys(api_key, server_api_key)


def verify_any_api_key(api_key: str, main_key: str, sub_keys: list) -> bool:
    """Verify an API key against the main key and all sub keys.

    Uses constant-time comparison for each key to prevent timing attacks.
    Checks the main key first, then iterates through sub keys.

    Args:
        api_key: The API key provided by the client.
        main_key: The server's main API key.
        sub_keys: List of SubKeyEntry objects with .key attribute.

    Returns:
        True if the API key matches any configured key, False otherwise.
    """
    if not api_key:
        return False
    # Check main key
    if main_key and compare_keys(api_key, main_key):
        return True
    # Check sub keys
    for sk in sub_keys:
        if sk.key and compare_keys(api_key, sk.key):
            return True
    return False


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Validate API key format requirements.

    Rules:
    - Minimum 4 characters
    - No whitespace characters (space, tab, newline, etc.)
    - Printable characters only (no control characters)
    - ASCII characters only

    The ASCII-only rule is not cosmetic: HTTP request headers are decoded as
    latin-1 by the ASGI layer, so a client cannot transmit a non-ASCII key
    intact. A configured key such as "café" therefore starts the server
    fine but can never be matched over the wire, yielding silent 401s on every
    authenticated request. Rejecting it at configuration time surfaces the
    misconfiguration immediately instead.

    Args:
        api_key: The API key string to validate.

    Returns:
        Tuple of (is_valid, error_message). Error message is empty if valid.
    """
    if len(api_key) < 4:
        return False, "API key must be at least 4 characters"
    if any(c.isspace() for c in api_key):
        return False, "API key must not contain whitespace"
    if not api_key.isprintable():
        return False, "API key must contain only printable characters"
    if not api_key.isascii():
        return False, "API key must contain only ASCII characters"
    return True, ""


def verify_session(request: Request) -> bool:
    """Verify if the request has a valid admin session.

    Checks for a valid session cookie in the request.

    Args:
        request: The FastAPI request object.

    Returns:
        True if the session is valid, False otherwise.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not isinstance(token, str) or not token:
        return False
    data = decode_session_token(token)
    if data is None:
        return False
    if data["auth_source"] == "trusted_network":
        return is_trusted_web_client(request)
    return True


def session_auth_source(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not isinstance(token, str) or not token:
        return None
    data = decode_session_token(token)
    if data is None:
        return None
    if data["auth_source"] == "trusted_network" and not is_trusted_web_client(request):
        return None
    return data["auth_source"]


async def require_admin(request: Request) -> bool:
    """FastAPI dependency to require admin authentication.

    This dependency can be used in route definitions to protect
    admin-only endpoints. It checks for a valid session cookie.

    Args:
        request: The FastAPI request object (injected by FastAPI).

    Returns:
        True if authentication is successful.

    Raises:
        HTTPException: 401 Unauthorized if not authenticated.

    Example:
        >>> from fastapi import Depends
        >>> @app.get("/admin/settings")
        ... async def get_settings(is_admin: bool = Depends(require_admin)):
        ...     return {"settings": "..."}
    """
    # Skip admin auth when skip_api_key_verification is enabled
    if _get_global_settings is not None:
        gs = _get_global_settings()
        if gs is not None and gs.auth.skip_api_key_verification:
            return True

    if not verify_session(request):
        # Browser requests (Accept: text/html) get redirected to login page
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            raise _RedirectToLogin()
        raise HTTPException(
            status_code=401,
            detail="Admin authentication required",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return True


class _RedirectToLogin(Exception):
    """Raised to trigger a redirect to the admin login page."""
    pass
