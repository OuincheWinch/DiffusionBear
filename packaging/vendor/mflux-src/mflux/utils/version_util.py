from __future__ import annotations

import importlib.metadata
from pathlib import Path

import toml


class VersionUtil:
    @staticmethod
    def get_mflux_version() -> str:
        return VersionUtil._scan_pyproject() or VersionUtil._get_installed_version() or "unknown"

    @staticmethod
    def _scan_pyproject(start: Path | None = None) -> str | None:
        # The source-checkout case: walking up from this file reaches mflux's own pyproject, whose
        # version is the truth even before a reinstall (the release script counts on that). The
        # walk stops at the first pyproject it meets, and only mflux's own counts: a copy installed
        # under a host project's .venv would otherwise reach the host's pyproject and report the
        # host's version as mflux's (#728). Anything else defers to the installed metadata.
        current_dir = (start or Path(__file__)).resolve().parent
        for parent in current_dir.parents:
            pyproject_path = parent / "pyproject.toml"
            if not pyproject_path.exists():
                continue
            try:
                project = toml.load(pyproject_path).get("project", {})
            except Exception:  # noqa: BLE001
                return None
            if project.get("name") != "mflux":
                return None
            return project.get("version")
        return None

    @staticmethod
    def _get_installed_version() -> str | None:
        try:
            return importlib.metadata.version("mflux")
        except importlib.metadata.PackageNotFoundError:
            return None
