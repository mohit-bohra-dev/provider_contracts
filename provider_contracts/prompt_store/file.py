"""File-based prompt store — reads markdown templates from a directory."""
from __future__ import annotations

from pathlib import Path

from ._base import AbstractPromptStoreProvider

_DEFAULT_BASE_DIR = "./docs"


class FilePromptStoreProvider(AbstractPromptStoreProvider):
    """Reads prompt templates from ``.md`` files in a base directory.

    Prompt names map to file paths: ``"agent.system"`` → ``<base_dir>/agent/system.md``.
    Dots in the name are treated as directory separators.
    """

    def __init__(self, base_dir: str = _DEFAULT_BASE_DIR) -> None:
        self._base_dir = Path(base_dir)

    def _resolve_path(self, name: str, version: str | None = None) -> Path:
        parts = name.split(".")
        if version:
            parts[-1] = f"{parts[-1]}_v{version}"
        return self._base_dir / "/".join(parts[:-1]) / f"{parts[-1]}.md"

    async def get(self, name: str, *, version: str | None = None) -> str:
        path = self._resolve_path(name, version)
        if not path.is_file():
            # Try without version if versioned file doesn't exist.
            if version:
                fallback = self._resolve_path(name, None)
                if fallback.is_file():
                    return fallback.read_text(encoding="utf-8")
            raise KeyError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    async def list_names(self) -> list[str]:
        if not self._base_dir.is_dir():
            return []
        names: list[str] = []
        for md_file in sorted(self._base_dir.rglob("*.md")):
            relative = md_file.relative_to(self._base_dir)
            # Convert path back to dot-separated name.
            name = ".".join(relative.with_suffix("").parts)
            names.append(name)
        return names
