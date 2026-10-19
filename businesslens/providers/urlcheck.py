"""Validation for user-supplied provider base URLs (SSRF guard).

Rules: https only, no embedded credentials, no query/fragment, a hostname (not
an IP literal), and every address the name resolves to must be a public
internet address. Loopback, private, link-local, reserved and cloud-metadata
ranges are refused. Redirects are disabled by the adapters.

Residual risk: the name is resolved here and again when connecting, so a
hostile DNS server could answer differently the second time (DNS rebinding).
For a hosted multi-user deployment, add an egress firewall as well.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

from businesslens.providers.base import ErrorKind, ProviderError

MAX_URL_CHARS = 200


def _bad(reason: str) -> ProviderError:
    return ProviderError(ErrorKind.INVALID_CONFIG, f"Base URL refused: {reason}.")


def validate_base_url(url: str, resolver=socket.getaddrinfo) -> str:
    """Return the normalised URL or raise ``ProviderError``. ``resolver`` is injectable for tests."""
    url = (url or "").strip()
    if not url or len(url) > MAX_URL_CHARS or not url.isascii() or any(c.isspace() for c in url):
        raise _bad("it is empty, too long or contains invalid characters")
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise _bad("only https:// is allowed")
    if parts.username or parts.password or "@" in parts.netloc:
        raise _bad("credentials in the URL are not allowed")
    if parts.query or parts.fragment:
        raise _bad("query strings and fragments are not allowed")
    host = parts.hostname
    try:
        port = parts.port
    except ValueError:
        raise _bad("invalid port") from None
    if not host:
        raise _bad("no hostname")
    if port not in (None, 443):
        raise _bad("only the default https port (443) is allowed")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise _bad("use a hostname, not an IP address")
    if host.lower() == "localhost" or host.lower().endswith((".local", ".internal", ".localhost")):
        raise _bad("local and internal hostnames are not allowed")
    try:
        infos = resolver(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise _bad("the hostname could not be resolved") from None
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise _bad("the hostname did not resolve")
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise _bad("it points to a non-public network address")
    return url.rstrip("/")
