"""SSRF egress allowlist for server-side URL fetches.

Every place this service dereferences a caller-influenced URL — the
``/media/import-url`` fetch, yt-dlp's generic extractor, and ffprobe's network
protocols — must first prove the URL's host resolves ONLY to public internet
addresses. This module is the single gate for that.

Threat model: an authenticated but UNTRUSTED tenant hands us a URL. A naive
fetch would let them reach the container's own loopback, the private
RFC1918/ULA network the container shares, the link-local range, or the cloud
metadata endpoint (``169.254.169.254``) — a classic SSRF pivot into
credentials.

Two traps this closes that a one-shot ``is_private`` check misses:

1. Redirects. httpx's ``follow_redirects=True`` re-resolves and re-connects
   with NO second check, so a public URL that 30x-redirects to ``127.0.0.1``
   sails through. ``guarded_stream`` follows redirects MANUALLY and validates
   every hop's target before connecting to it.

2. DNS rebinding / TOCTOU. A hostname can resolve to a public address for the
   check and a private one for the connect. We resolve ONCE, validate the
   resolved IPs, and PIN the connection to a validated IP (the TLS SNI and
   certificate verification still use the real hostname), so the socket lands
   on exactly the address we vetted — a second lookup cannot swap it.

The import path gets the full pinned ``guarded_stream``. yt-dlp and ffprobe
run the socket themselves, so for those we can only pre-flight the host with
``assert_url_egress_allowed`` (a best-effort check that cannot pin the eventual
connection); see the call sites for that documented limitation.
"""
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit

import httpx

from app.config import get_settings
from app.editor.errors import EditorError

IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
BlockPredicate = Callable[[IpAddress], bool]

# Named for intent even though it also falls under ``is_link_local``: the cloud
# metadata endpoint is THE SSRF prize, so it earns an explicit mention.
CLOUD_METADATA_IPV4 = ipaddress.ip_address("169.254.169.254")

# A hop budget: enough for a legitimate CDN's canonicalising redirect chain,
# small enough that a redirect loop cannot spin us forever.
MAX_REDIRECTS = 5

# The schemes we are willing to dereference server-side. Anything else
# (``file:``, ``gopher:``, ``ftp:`` …) is an SSRF vector in its own right.
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

# Streaming read size for the import path — matches ``uploads.CHUNK_BYTES`` in
# spirit: large enough to avoid a syscall storm, small enough that the in-RAM
# working set stays a single chunk rather than the whole body.
STREAM_CHUNK_BYTES = 64 * 1024

# Non-public ranges that ``ipaddress``'s ``is_private`` does NOT flag on every
# Python version — carrier-grade NAT (RFC6598) and IPv4-mapped IPv6, either of
# which can still land a fetch on internal infrastructure. Belt-and-braces on
# top of the ``is_*`` predicates below.
_EXTRA_BLOCKED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),   # CGNAT (RFC6598)
    ipaddress.ip_network("::ffff:0:0/96"),   # IPv4-mapped IPv6
)


class EgressBlockedError(EditorError):
    """A URL was refused because its host resolves to a non-public address, or
    could not be resolved/validated at all.

    Subclasses ``EditorError`` so the HTTP boundary maps it to 422 alongside
    the other "your request is the problem" failures — reaching a private
    address is a client error, not a server fault.
    """


def is_blocked_ip(ip: IpAddress) -> bool:
    """True when ``ip`` is anything other than a public, globally-routable
    address: loopback, private (RFC1918 / CGNAT / ULA), link-local (incl. the
    metadata endpoint), multicast, reserved, or unspecified.

    ``EGRESS_ALLOWED_NETWORKS`` (empty by default) can exempt specific CIDRs —
    a developer-machine escape hatch for fake-ip VPN resolvers, documented on
    ``app.config.Settings.egress_allowed_networks``. It is consulted FIRST and
    only ever un-blocks; nothing in it can make a public address blocked.
    """
    if any(ip in network for network in get_settings().egress_allowed_ip_networks):
        return False
    return (
        ip.is_private          # RFC1918, 127/8, 169.254/16, ULA, TEST-NET…
        or ip.is_loopback
        or ip.is_link_local    # incl. 169.254.169.254 metadata
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or any(ip in network for network in _EXTRA_BLOCKED_NETWORKS)
    )


def _resolve_addresses(host: str, port: int) -> list[IpAddress]:
    """Every IP ``host`` resolves to right now, as parsed ``ip_address``es.

    A literal IP resolves to itself. A resolution failure is a hard block, not
    a pass-through — we never fetch a host we could not vet.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise EgressBlockedError(f"could not resolve host {host!r}") from exc
    addresses: list[IpAddress] = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        raw = sockaddr[0]
        # Strip any IPv6 zone id (``fe80::1%eth0``) before parsing.
        ip_text = raw.split("%", 1)[0]
        addresses.append(ipaddress.ip_address(ip_text))
    return addresses


def _host_and_port(url: str) -> tuple[str, int, str]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise EgressBlockedError(f"refusing URL scheme {scheme!r}: {url!r}")
    host = parts.hostname
    if not host:
        raise EgressBlockedError(f"URL has no host: {url!r}")
    port = parts.port or _DEFAULT_PORTS[scheme]
    return host, port, scheme


def resolve_allowed_addresses(
    host: str, port: int, *, is_blocked: BlockPredicate = is_blocked_ip
) -> list[IpAddress]:
    """Resolve ``host`` and return its addresses, or raise ``EgressBlockedError``
    if it does not resolve or ANY resolved address is non-public.

    All-or-nothing on purpose: a name that resolves to one public and one
    private address is refused, since which one a later connect() would pick is
    not ours to gamble on.
    """
    addresses = _resolve_addresses(host, port)
    if not addresses:
        raise EgressBlockedError(f"host {host!r} did not resolve to any address")
    for ip in addresses:
        if is_blocked(ip):
            raise EgressBlockedError(
                f"refusing to fetch {host!r}: resolves to non-public address {ip}"
            )
    return addresses


def assert_url_egress_allowed(
    url: str, *, is_blocked: BlockPredicate = is_blocked_ip
) -> None:
    """Pre-flight egress check for fetchers whose socket we do NOT own (yt-dlp,
    ffprobe): resolve the URL's host and reject if it is non-public.

    Best-effort by construction — it validates the host as it resolves *now*
    and cannot pin the connection the external tool subsequently makes, so it
    does not close the DNS-rebind window the way ``guarded_stream`` does. It is
    still the right gate for those paths: it stops the obvious cases (private
    literals, metadata, a host that only resolves private) before the tool runs.
    """
    host, port, _scheme = _host_and_port(url)
    resolve_allowed_addresses(host, port, is_blocked=is_blocked)


def _authority_header(host: str, port: int, scheme: str) -> str:
    """The ``Host`` header value for the real host, since the request URL we
    build carries the pinned IP instead."""
    if port == _DEFAULT_PORTS[scheme]:
        return host
    return f"{host}:{port}"


def _open_validated_hop(
    client: httpx.Client, url: str, is_blocked: BlockPredicate
) -> httpx.Response:
    """Validate ``url``'s host, then open a streaming GET pinned to a validated
    IP (real host preserved for the ``Host`` header and TLS SNI/verification)."""
    host, port, scheme = _host_and_port(url)
    addresses = resolve_allowed_addresses(host, port, is_blocked=is_blocked)
    pinned_ip = str(addresses[0])
    pinned_url = httpx.URL(url).copy_with(host=pinned_ip)
    headers = {"Host": _authority_header(host, port, scheme)}
    extensions = {"sni_hostname": host} if scheme == "https" else {}
    request = client.build_request("GET", pinned_url, headers=headers, extensions=extensions)
    return client.send(request, stream=True)


@contextmanager
def guarded_stream(
    url: str,
    *,
    timeout: float,
    max_redirects: int = MAX_REDIRECTS,
    is_blocked: BlockPredicate = is_blocked_ip,
) -> Iterator[httpx.Response]:
    """Yield a streaming ``httpx.Response`` for ``url`` with SSRF egress guards.

    Redirects are followed MANUALLY: each hop's target is resolved and
    validated, and the connection is pinned to a validated IP, so neither a
    redirect to a private address nor a DNS-rebind between check and connect
    can smuggle the socket onto an internal host. The caller consumes the body
    via ``response.iter_bytes()``; the response is closed on exit.

    ``trust_env=False`` is a security requirement, not a tidiness one: an
    ambient ``HTTP(S)_PROXY``/``ALL_PROXY`` would tunnel the request through a
    proxy that re-resolves the hostname, defeating the IP pin entirely.
    """
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        current = url
        for _hop in range(max_redirects + 1):
            response = _open_validated_hop(client, current, is_blocked)
            if response.is_redirect:
                location = response.headers.get("location", "")
                response.close()
                if not location:
                    raise EgressBlockedError("redirect response carried no Location header")
                current = str(httpx.URL(current).join(location))
                continue
            try:
                yield response
            finally:
                response.close()
            return
        raise EgressBlockedError(f"too many redirects (exceeded {max_redirects})")
