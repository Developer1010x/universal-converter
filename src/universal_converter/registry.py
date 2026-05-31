"""Converter registry and format discovery.

This module provides a lightweight, dependency-free way to discover which
conversions are available and to look up the converter that handles a given
source/target format pair. It complements the lazy-loading design of the
package: importing this module does NOT import heavy optional dependencies,
since converter classes are inspected only for their declared
``SUPPORTED_CONVERSIONS`` metadata.

Example:
    >>> from universal_converter.registry import ConverterRegistry
    >>> registry = ConverterRegistry()
    >>> registry.can_convert('json', 'csv')
    True
    >>> all(t in registry.targets_for('json') for t in ('csv', 'html', 'xml'))
    True
    >>> registry.detect_format('data.JSON')
    'json'
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

from .converters import BaseConverter

# Common file extensions whose canonical format name differs from the
# extension string. Anything not listed maps to its lowercased extension.
_EXTENSION_ALIASES: Dict[str, str] = {
    "yml": "yaml",
    "htm": "html",
    "jpeg": "jpg",
    "tif": "tiff",
    "markdown": "md",
    "text": "txt",
}


class ConverterRegistry:
    """Discovers converter classes and answers capability questions.

    The registry walks the ``universal_converter.converters`` package, imports
    each submodule, and collects every concrete :class:`BaseConverter`
    subclass. Discovery happens once on first use and is cached. Modules that
    fail to import (for example because an optional dependency is missing) are
    skipped silently so that ``import``-time failures never break discovery.
    """

    def __init__(self) -> None:
        self._converters: Optional[List[Type[BaseConverter]]] = None

    def _discover(self) -> List[Type[BaseConverter]]:
        if self._converters is not None:
            return self._converters

        found: List[Type[BaseConverter]] = []
        package_name = f"{__package__}.converters"
        package = importlib.import_module(package_name)
        package_dir = Path(package.__file__).parent

        for module_info in pkgutil.iter_modules([str(package_dir)]):
            full_name = f"{package_name}.{module_info.name}"
            try:
                module = importlib.import_module(full_name)
            except Exception:
                # Optional dependency missing or module-level error: skip it.
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseConverter)
                    and obj is not BaseConverter
                    and obj.__module__ == full_name
                    and getattr(obj, "SUPPORTED_CONVERSIONS", None)
                ):
                    found.append(obj)

        # Stable, priority-aware ordering (higher PRIORITY first).
        found.sort(key=lambda c: (-getattr(c, "PRIORITY", 50), c.__name__))
        self._converters = found
        return found

    @property
    def converters(self) -> List[Type[BaseConverter]]:
        """All discovered concrete converter classes."""
        return list(self._discover())

    def find_converter(
        self, source_format: str, target_format: str
    ) -> Optional[Type[BaseConverter]]:
        """Return the highest-priority converter class for a format pair.

        Returns ``None`` if no registered converter declares support for the
        conversion. Format names are matched case-insensitively.
        """
        src = source_format.lower().lstrip(".")
        tgt = target_format.lower().lstrip(".")
        for converter in self._discover():
            if tgt in converter.SUPPORTED_CONVERSIONS.get(src, []):
                return converter
        return None

    def can_convert(self, source_format: str, target_format: str) -> bool:
        """Return ``True`` if any converter handles the format pair."""
        return self.find_converter(source_format, target_format) is not None

    def targets_for(self, source_format: str) -> List[str]:
        """List every target format reachable from ``source_format``."""
        src = source_format.lower().lstrip(".")
        targets: set = set()
        for converter in self._discover():
            targets.update(converter.SUPPORTED_CONVERSIONS.get(src, []))
        return sorted(targets)

    def supported_conversions(self) -> Dict[str, List[str]]:
        """Return a merged ``{source: [targets...]}`` map across converters."""
        merged: Dict[str, set] = {}
        for converter in self._discover():
            for src, targets in converter.SUPPORTED_CONVERSIONS.items():
                merged.setdefault(src, set()).update(targets)
        return {src: sorted(t) for src, t in sorted(merged.items())}

    @staticmethod
    def detect_format(path: str) -> Optional[str]:
        """Infer the canonical format name from a file path/extension.

        Returns the normalized format string (lowercased, alias-resolved) or
        ``None`` when the path has no usable extension.
        """
        suffix = Path(path).suffix.lower().lstrip(".")
        if not suffix:
            return None
        return _EXTENSION_ALIASES.get(suffix, suffix)


# Module-level convenience singleton and thin wrappers.
_default_registry = ConverterRegistry()


def find_converter(source_format: str, target_format: str):
    """Module-level shortcut for :meth:`ConverterRegistry.find_converter`."""
    return _default_registry.find_converter(source_format, target_format)


def can_convert(source_format: str, target_format: str) -> bool:
    """Module-level shortcut for :meth:`ConverterRegistry.can_convert`."""
    return _default_registry.can_convert(source_format, target_format)


def list_conversions() -> Dict[str, List[str]]:
    """Module-level shortcut for :meth:`ConverterRegistry.supported_conversions`."""
    return _default_registry.supported_conversions()


def detect_format(path: str) -> Optional[str]:
    """Module-level shortcut for :meth:`ConverterRegistry.detect_format`."""
    return ConverterRegistry.detect_format(path)
