"""Lightweight sanity checks — verify wiring without running real generation.

Mirrors the reference repo's CheckResult pattern: each check either passes
or fails with a message, and one failing check doesn't stop the rest.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str
