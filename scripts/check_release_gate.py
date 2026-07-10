"""Release-version guardrails for the unified somm workspace.

This script intentionally has no third-party dependencies so it can run in CI
before build or publish steps. It enforces two invariants:

1. Every workspace package uses the same version and matching inter-package
   dependency pins.
2. Major-version releases are blocked until the repository carries an explicit
   1.0 approval note.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVAL_PATH = ROOT / "notes" / "ONE_DOT_ZERO_GO_DECISION.md"
APPROVAL_MARKER = "somm-1.0-go: true"

PYPROJECTS = [
    ROOT / "pyproject.toml",
    ROOT / "packages" / "somm" / "pyproject.toml",
    ROOT / "packages" / "somm-core" / "pyproject.toml",
    ROOT / "packages" / "somm-service" / "pyproject.toml",
    ROOT / "packages" / "somm-mcp" / "pyproject.toml",
    ROOT / "packages" / "somm-langchain" / "pyproject.toml",
    ROOT / "packages" / "somm-skill" / "pyproject.toml",
]


def _load_project(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)["project"]


def _core_version() -> str:
    version_py = ROOT / "packages" / "somm-core" / "src" / "somm_core" / "version.py"
    match = re.search(r'^VERSION = "([^"]+)"', version_py.read_text(), re.M)
    if not match:
        raise RuntimeError("could not find VERSION in somm_core/version.py")
    return match.group(1)


def _major(version: str) -> int:
    match = re.match(r"^(\d+)\.\d+\.\d+(?:[.-].*)?$", version)
    if not match:
        raise RuntimeError(f"unsupported version format: {version!r}")
    return int(match.group(1))


def main() -> int:
    errors: list[str] = []
    versions: dict[str, str] = {}
    for path in PYPROJECTS:
        rel = path.relative_to(ROOT)
        project = _load_project(path)
        versions[str(rel)] = project["version"]

    root_version = versions["pyproject.toml"]
    for rel, version in versions.items():
        if version != root_version:
            errors.append(f"{rel} version {version} != workspace version {root_version}")

    core_version = _core_version()
    if core_version != root_version:
        errors.append(f"somm_core.VERSION {core_version} != workspace version {root_version}")

    for path in PYPROJECTS:
        project = _load_project(path)
        for dep in project.get("dependencies", []):
            if dep.startswith(("somm==", "somm-core==")) and not dep.endswith(
                f"=={root_version}"
            ):
                errors.append(
                    f"{path.relative_to(ROOT)} dependency pin {dep!r} "
                    f"does not match {root_version}"
                )

    if _major(root_version) >= 1:
        approved = (
            APPROVAL_PATH.exists()
            and APPROVAL_MARKER in APPROVAL_PATH.read_text(encoding="utf-8")
        )
        if not approved:
            errors.append(
                "major-version release blocked: create "
                f"{APPROVAL_PATH.relative_to(ROOT)} containing {APPROVAL_MARKER!r} "
                "only after a deliberate 1.0 readiness decision"
            )

    if errors:
        for error in errors:
            print(f"release-gate: {error}", file=sys.stderr)
        return 1
    print(f"release-gate: ok ({root_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
