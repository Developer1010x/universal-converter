"""Converter registry, capability lookup and multi-hop route planning.

Importing this module does not pull in any optional dependency: converter
modules keep their heavy imports inside methods, so discovery only reads the
declared ``SUPPORTED_CONVERSIONS`` metadata off each class.

Example:
    >>> from universal_converter.registry import ConverterRegistry
    >>> registry = ConverterRegistry()
    >>> registry.can_convert('json', 'csv')
    True
    >>> all(t in registry.targets_for('json') for t in ('csv', 'html', 'xml'))
    True
    >>> registry.detect_format('data.JSON')
    'json'
    >>> [step.target for step in registry.find_route('csv', 'pdf')]
    ['html', 'pdf']
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections import deque
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type

from .converters import BaseConverter

# Common file extensions whose canonical format name differs from the
# extension string. Anything not listed maps to its lowercased extension.
_EXTENSION_ALIASES: Dict[str, str] = {
    "yml": "yaml",
    "htm": "html",
    "jpeg": "jpg",
    "tif": "tiff",
    "markdown": "md",
    "mdown": "md",
    "text": "txt",
    "sqlite3": "sqlite",
    "fa": "fasta",
    "fna": "fasta",
    "fq": "fastq",
    "gb": "genbank",
    "gbk": "genbank",
}


class RouteStep(NamedTuple):
    """One hop of a conversion route."""

    source: str
    target: str
    converter: Type[BaseConverter]

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.source} -> {self.target} ({self.converter.__name__})"


class ConverterRegistry:
    """Discovers converter classes and answers capability questions.

    The registry walks the ``universal_converter.converters`` package, imports
    each submodule, and collects every concrete :class:`BaseConverter`
    subclass. Discovery happens once on first use and is cached.

    Converters are ordered by ``PRIORITY`` ascending -- **lower wins** -- so a
    specialist (``ImageConverter``, 10) always beats the generic fallback
    (``DataConverter``, 50) for a pair both of them declare.
    """

    def __init__(self) -> None:
        self._converters: Optional[List[Type[BaseConverter]]] = None
        self._import_errors: Dict[str, str] = {}

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
            except Exception as exc:
                # Recorded rather than swallowed: `--doctor` reports these, so
                # a converter can no longer disappear from --list in silence.
                self._import_errors[full_name] = f"{type(exc).__name__}: {exc}"
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseConverter)
                    and obj is not BaseConverter
                    and obj.__module__ == full_name
                    and getattr(obj, "SUPPORTED_CONVERSIONS", None)
                ):
                    found.append(obj)

        found.sort(key=lambda c: (getattr(c, "PRIORITY", 50), c.__name__))
        self._converters = found
        return found

    @property
    def converters(self) -> List[Type[BaseConverter]]:
        """All discovered concrete converter classes, best priority first."""
        return list(self._discover())

    @property
    def import_errors(self) -> Dict[str, str]:
        """Modules that failed to import during discovery, with the reason."""
        self._discover()
        return dict(self._import_errors)

    def find_converter(
        self, source_format: str, target_format: str
    ) -> Optional[Type[BaseConverter]]:
        """Return the highest-priority converter class for a format pair.

        Returns ``None`` if no registered converter declares the conversion.
        Format names are normalised (case, leading dot, aliases) first.
        """
        src = self.normalize(source_format)
        tgt = self.normalize(target_format)
        for converter in self._discover():
            if tgt in converter.SUPPORTED_CONVERSIONS.get(src, []):
                return converter
        return None

    def can_convert(self, source_format: str, target_format: str) -> bool:
        """Return ``True`` if any converter handles the format pair directly."""
        return self.find_converter(source_format, target_format) is not None

    def targets_for(self, source_format: str) -> List[str]:
        """List every target format reachable from ``source_format`` in one hop."""
        src = self.normalize(source_format)
        targets: Set[str] = set()
        for converter in self._discover():
            targets.update(converter.SUPPORTED_CONVERSIONS.get(src, []))
        return sorted(targets)

    def supported_conversions(self) -> Dict[str, List[str]]:
        """Return a merged ``{source: [targets...]}`` map across converters."""
        merged: Dict[str, Set[str]] = {}
        for converter in self._discover():
            for src, targets in converter.SUPPORTED_CONVERSIONS.items():
                merged.setdefault(src, set()).update(targets)
        return {src: sorted(t) for src, t in sorted(merged.items())}

    def pair_count(self) -> int:
        """Total number of distinct source/target pairs in the registry."""
        return sum(len(t) for t in self.supported_conversions().values())

    def formats(self) -> List[str]:
        """Every format name the registry knows, as a source or a target."""
        conversions = self.supported_conversions()
        names: Set[str] = set(conversions)
        for targets in conversions.values():
            names.update(targets)
        return sorted(names)

    def routed_pair_count(self, max_hops: int = 3) -> int:
        """Pairs reachable only by chaining converters, excluding direct ones."""
        return sum(
            1
            for source in self.formats()
            for target in self.reachable_from(source, max_hops=max_hops)
            if not self.can_convert(source, target)
        )

    # --- multi-hop routing ----------------------------------------------

    def find_route(
        self,
        source_format: str,
        target_format: str,
        max_hops: int = 3,
        require_available: bool = False,
    ) -> Optional[List[RouteStep]]:
        """Return the shortest chain of conversions from source to target.

        Breadth-first search over the registry's ``{source: [targets]}``
        adjacency map, so the first route found is the one with the fewest
        hops. A direct conversion comes back as a single-step route.

        With ``require_available`` the search only crosses edges whose
        converter has all of its dependencies installed, so routing never
        proposes a path that is going to fail on a missing import.
        """
        src = self.normalize(source_format)
        tgt = self.normalize(target_format)
        if src == tgt:
            return []

        graph = self.supported_conversions()
        # (format, path-so-far); BFS guarantees the first hit is shortest.
        queue: deque[Tuple[str, List[RouteStep]]] = deque([(src, [])])
        seen: Set[str] = {src}

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_hops:
                continue

            for candidate in graph.get(current, []):
                if candidate in seen:
                    continue
                converter = self.find_converter(current, candidate)
                if converter is None:
                    continue
                if require_available and converter.missing_for(current, candidate):
                    continue

                step_path = path + [RouteStep(current, candidate, converter)]
                if candidate == tgt:
                    return step_path
                seen.add(candidate)
                queue.append((candidate, step_path))

        return None

    def reachable_from(self, source_format: str, max_hops: int = 3) -> Dict[str, int]:
        """Map every format reachable from ``source_format`` to its hop count."""
        src = self.normalize(source_format)
        graph = self.supported_conversions()
        distances: Dict[str, int] = {}
        queue: deque[Tuple[str, int]] = deque([(src, 0)])
        seen = {src}

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for candidate in graph.get(current, []):
                if candidate in seen:
                    continue
                seen.add(candidate)
                distances[candidate] = depth + 1
                queue.append((candidate, depth + 1))
        return distances

    # --- format names -----------------------------------------------------

    @staticmethod
    def normalize(format_name: str) -> str:
        """Lowercase, strip a leading dot and resolve extension aliases."""
        name = (format_name or "").lower().lstrip(".")
        return _EXTENSION_ALIASES.get(name, name)

    @classmethod
    def detect_format(cls, path: str) -> Optional[str]:
        """Infer the canonical format name from a file path/extension.

        Returns the normalised format string or ``None`` when the path carries
        no usable extension. Dotfiles such as ``.env`` are read as their name.
        """
        candidate = Path(path)
        suffix = candidate.suffix.lower().lstrip(".")
        if not suffix:
            # ".env" has no suffix but its name is the format.
            bare = candidate.name.lower().lstrip(".")
            return cls.normalize(bare) if bare else None
        return cls.normalize(suffix)


# Module-level convenience singleton and thin wrappers.
_default_registry = ConverterRegistry()


def get_registry() -> ConverterRegistry:
    """Return the process-wide registry singleton."""
    return _default_registry


def find_converter(source_format: str, target_format: str):
    """Module-level shortcut for :meth:`ConverterRegistry.find_converter`."""
    return _default_registry.find_converter(source_format, target_format)


def can_convert(source_format: str, target_format: str) -> bool:
    """Module-level shortcut for :meth:`ConverterRegistry.can_convert`."""
    return _default_registry.can_convert(source_format, target_format)


def list_conversions() -> Dict[str, List[str]]:
    """Module-level shortcut for :meth:`ConverterRegistry.supported_conversions`."""
    return _default_registry.supported_conversions()


def find_route(
    source_format: str,
    target_format: str,
    max_hops: int = 3,
    require_available: bool = False,
) -> Optional[List[RouteStep]]:
    """Module-level shortcut for :meth:`ConverterRegistry.find_route`."""
    return _default_registry.find_route(
        source_format, target_format, max_hops, require_available
    )


def detect_format(path: str) -> Optional[str]:
    """Module-level shortcut for :meth:`ConverterRegistry.detect_format`."""
    return ConverterRegistry.detect_format(path)


def format_route(steps: Sequence[RouteStep]) -> str:
    """Render a route as ``json -> html -> md`` for display."""
    if not steps:
        return ""
    return " -> ".join([steps[0].source] + [step.target for step in steps])
