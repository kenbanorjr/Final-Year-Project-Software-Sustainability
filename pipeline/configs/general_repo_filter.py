"""
Shared SonarQube inclusion/exclusion patterns for scanning and mining.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable

SONAR_INCLUSIONS = (
    "**/*.py",
    "**/*.java",
    "**/*.kt",
    "**/*.go",
    "**/*.ts",
    "**/*.js",
)

SONAR_EXCLUSIONS = (
    "**/tests/**",
    "**/test/**",
    "**/__tests__/**",
    "**/*_test.go",
    "**/*.spec.ts",
    "**/*.spec.js",
    "**/*.test.ts",
    "**/*.test.js",
    "**/*Test.java",
    "**/*Tests.java",
    "**/node_modules/**",
    "**/vendor/**",
    "**/third_party/**",
    "**/dist/**",
    "**/build/**",
    "**/out/**",
    "**/target/**",
    "**/bin/**",
    "**/obj/**",
    "**/.gradle/**",
    "**/.mvn/**",
    "**/.venv/**",
    "**/venv/**",
    "**/env/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.min.js",
    "**/*.bundle.js",
    "**/*.map",
    "**/*.d.ts",
    "**/generated/**",
    "**/*.pb.go",
    "**/*_gen.go",
    "**/*_generated.go",
    "**/docs/**",
    "**/doc/**",
    "**/site/**",
    "**/website/**",
    "**/public/**",
    "**/__init__.py",
    "**/static/**",
    "**/.github/**",
    "**/.gitlab/**",
    "**/.circleci/**",
    "**/examples/**",
    "**/example/**",
    "**/demo/**",
    "**/demos/**",
    "**/scripts/**",
    "**/tools/**",
    "**/tooling/**",
    "**/bench/**",
    "**/benchmark/**",
    "**/benchmarks/**",
)


def _normalize_path(path: str | PurePosixPath) -> str:
    return str(path).replace("\\", "/")


def _pattern_variants(pattern: str) -> tuple[str, ...]:
    """
    Return pattern variants for robust top-level matching.

    `PurePosixPath.match('**/*.js')` does not match `index.js`.
    For patterns beginning with `**/`, also test the stripped variant.
    """
    if pattern.startswith("**/"):
        stripped = pattern[3:]
        return (pattern, stripped)
    return (pattern,)


def matches_patterns(path: str | PurePosixPath, patterns: Iterable[str]) -> bool:
    normalized = _normalize_path(path)
    posix_path = PurePosixPath(normalized)
    for pattern in patterns:
        for variant in _pattern_variants(pattern):
            if variant and posix_path.match(variant):
                return True
    return False


def is_included_path(path: str | PurePosixPath) -> bool:
    return matches_patterns(path, SONAR_INCLUSIONS)


def is_excluded_path(path: str | PurePosixPath) -> bool:
    return matches_patterns(path, SONAR_EXCLUSIONS)


def included_extensions() -> frozenset[str]:
    exts = {PurePosixPath(pattern).suffix.lower() for pattern in SONAR_INCLUSIONS}
    return frozenset(exts)
