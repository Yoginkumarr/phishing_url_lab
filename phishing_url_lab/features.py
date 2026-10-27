"""Versioned, URL-only features used identically by training and Flask.

Intentional safeguards for a biased training corpus:
- Remove a leading www. before computing hostname statistics.
- Ignore URL scheme, paths, queries, and fragments entirely.
- Do not use PhiUSIIL's precomputed, label-correlated URLSimilarityIndex.

Limitation: hostname-only features cannot recognize attacks that live solely in
paths on otherwise legitimate domains. No website is visited by this module.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import tldextract

FEATURE_VERSION = "hostname_safe_v1"
FEATURE_NAMES = [
    "domain_label_length", "domain_digit_count", "domain_digit_ratio",
    "domain_hyphen_count", "domain_entropy", "domain_vowel_ratio",
    "tld_length", "host_length", "host_digit_count", "host_digit_ratio",
    "host_hyphen_count", "host_alpha_ratio", "host_entropy",
    "host_longest_digit_run", "host_longest_letter_run", "host_has_punycode",
    "host_is_ip",
]

# Uses packaged Public Suffix List snapshot. No background network requests.
_PSL = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _url_parts(raw_url: str):
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("Provide a nonempty URL string.")
    raw = raw_url.strip()
    if len(raw) > 8192:
        raise ValueError("URL is too long (maximum: 8192 characters).")
    if any(ch.isspace() for ch in raw):
        raise ValueError("URLs containing unescaped whitespace are not supported.")
    if raw.startswith("//"):
        raw = "http:" + raw
    elif not _SCHEME.match(raw):
        # Reject strings like javascript:alert(1), mailto:, or data:.
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw):
            raise ValueError("Only HTTP and HTTPS URLs are supported.")
        raw = "http://" + raw
    try:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only HTTP and HTTPS URLs are supported.")
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname or hostname.startswith("."):
            raise ValueError("URL is missing a valid hostname.")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid URL: " + str(exc)) from exc
    # Strip www to prevent a shortcut inherited from PhiUSIIL.
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def registered_domain_from_url(url: str) -> str:
    host = _url_parts(url)
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    parts = _PSL(host)
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}"
    return host


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return float(-sum((count / n) * math.log2(count / n)
                      for count in counts.values()))


def _longest_run(pattern: str, text: str) -> int:
    return max((len(m.group()) for m in re.finditer(pattern, text)), default=0)


def extract_lexical_features(url: str) -> dict[str, float]:
    """Return deterministic numeric features; never fetches the supplied URL."""
    host = _url_parts(url)
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0
    ext = _PSL(host) if not is_ip else None
    domain = ext.domain if ext and ext.domain else host
    suffix = ext.suffix if ext else ""

    def digits(s):
        return sum(ch.isdigit() for ch in s)
    def letters(s):
        return sum(ch.isalpha() for ch in s)

    host_len = max(len(host), 1)
    dom_len = max(len(domain), 1)
    host_digits = digits(host)
    domain_digits = digits(domain)
    result = {
        "domain_label_length": len(domain),
        "domain_digit_count": domain_digits,
        "domain_digit_ratio": domain_digits / dom_len,
        "domain_hyphen_count": domain.count("-"),
        "domain_entropy": _entropy(domain),
        "domain_vowel_ratio": sum(c in "aeiou" for c in domain) / dom_len,
        "tld_length": len(suffix),
        "host_length": len(host),
        "host_digit_count": host_digits,
        "host_digit_ratio": host_digits / host_len,
        "host_hyphen_count": host.count("-"),
        "host_alpha_ratio": letters(host) / host_len,
        "host_entropy": _entropy(host),
        "host_longest_digit_run": _longest_run(r"[0-9]+", host),
        "host_longest_letter_run": _longest_run(r"[a-z]+", host),
        "host_has_punycode": int("xn--" in host),
        "host_is_ip": is_ip,
    }
    return {name: float(result[name]) for name in FEATURE_NAMES}
