#!/usr/bin/env python3
"""Heuristic Swift risk scanner.

This script finds review candidates. It does not declare bugs. Every match must be
validated against business semantics and the complete runtime path.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    note: str


PATTERNS = [
    Pattern("try-optional", re.compile(r"\btry\?\b"), "May hide a business or persistence failure."),
    Pattern("empty-fallback", re.compile(r"\?\?\s*\[\s*\]"), "May merge query failure with valid empty data."),
    Pattern("empty-catch", re.compile(r"catch\s*\{\s*\}"), "Error is swallowed."),
    Pattern("print-error", re.compile(r"\bprint\s*\("), "Printing alone is usually insufficient error handling."),
    Pattern("task-sleep", re.compile(r"Task\.sleep"), "May indicate timing-based synchronization or flaky tests."),
    Pattern("userdefaults-standard", re.compile(r"UserDefaults\.standard"), "Shared global state needs review, especially in tests."),
    Pattern("current-date", re.compile(r"Date\s*\(\s*\)"), "Business time may need injection."),
    Pattern("onappear", re.compile(r"\.onAppear\b"), "Check reload ownership and repeated side effects."),
    Pattern("onchange", re.compile(r"\.onChange\b"), "Check feedback loops and duplicate work."),
    Pattern("assertion-continue", re.compile(r"assertionFailure\s*\("), "Ensure production does not continue with invalid facts."),
]

EXCLUDED_DIRS = {".git", ".build", "DerivedData", "Pods", "Carthage", "build"}


def iter_swift_files(root: Path):
    for path in root.rglob("*.swift"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="Project directory")
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    count = 0
    for path in iter_swift_files(root):
        relative = path.relative_to(root)
        if args.tests_only and not any("test" in part.lower() for part in relative.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            for pattern in PATTERNS:
                if pattern.regex.search(line):
                    count += 1
                    print(f"{relative}:{number}: [{pattern.name}] {line.strip()}")
                    print(f"  note: {pattern.note}")
    print(f"\nCandidates: {count}. Review manually; matches are not confirmed defects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
