"""Cluster detection: SALVO_CLUSTER > CC_CLUSTER > hostname patterns."""

from __future__ import annotations

import os
import re
import socket

_HOSTNAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"mila.*\.iro\.umontreal\.ca$"
            r"|mila-(?:cluster-)?login"
            r"|^login-?\d+\.server\.mila\.quebec$"
            r"|^cn-[a-z]\d+(?:\.|$)"
            r"|^mila-(?:l40s|cpu|gpu)-?\d+"
        ),
        "mila",
    ),
    # NOTE: nc*/ng*/blg*/cdr* compute hostnames are ambiguous across DRAC
    # clusters (narval vs rorqual share nc*/ng*); rely on CC_CLUSTER env on
    # DRAC compute nodes instead of pattern-matching bare compute hostnames.
    (re.compile(r"^login\d*\.rorqual\."), "rorqual"),
    (re.compile(r"^login\d*\.narval\."), "narval"),
    (re.compile(r"^login\d*\.beluga\."), "beluga"),
    (re.compile(r"^login\d*\.cedar\."), "cedar"),
]


def detect_cluster() -> str | None:
    if v := os.environ.get("SALVO_CLUSTER"):
        return v
    if v := os.environ.get("CC_CLUSTER"):
        return v
    host = socket.gethostname()
    for pattern, cluster_id in _HOSTNAME_PATTERNS:
        if pattern.search(host):
            return cluster_id
    return None
