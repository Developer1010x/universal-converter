"""Base converter interface plus lazy access to every concrete converter.

Concrete converter classes are resolved through :pep:`562` module ``__getattr__``
so that ``from universal_converter.converters import DataConverter`` works
without importing every converter module (and therefore every optional
dependency) up front.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional


@dataclass
class ConversionTask:
    """A single source -> target conversion request."""

    source_path: Path
    target_path: Path
    source_format: str
    target_format: str
    options: Dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))


@dataclass
class ConversionResult:
    """Outcome of a conversion attempt."""

    success: bool
    data: Any = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseConverter(ABC):
    """Base class for all converters.

    Subclasses declare the pairs they implement in ``SUPPORTED_CONVERSIONS``
    and are discovered automatically by :mod:`universal_converter.registry`.

    ``PRIORITY`` orders competing converters: **lower wins**. Specialised
    converters use small numbers (images = 10) and the generic data converter
    keeps the default 50, so a specialist always beats the fallback.
    """

    SUPPORTED_CONVERSIONS: ClassVar[Dict[str, List[str]]] = {}
    PRIORITY: ClassVar[int] = 50
    #: External binaries that must be on PATH.
    REQUIRES_EXTERNAL: ClassVar[List[str]] = []
    #: Importable Python module names required by at least one pair.
    REQUIRES_PYTHON: ClassVar[List[str]] = []

    @abstractmethod
    def convert(self, task: ConversionTask) -> ConversionResult:
        """Perform the conversion described by ``task``."""

    def can_handle(self, source_fmt: str, target_fmt: str) -> bool:
        """Return ``True`` if this converter declares the given pair."""
        return target_fmt.lower() in self.SUPPORTED_CONVERSIONS.get(
            source_fmt.lower(), []
        )

    @classmethod
    def requirements_for(cls, source_format: str, target_format: str) -> List[str]:
        """Requirements for one pair. Override when they vary per pair."""
        return list(cls.REQUIRES_EXTERNAL) + list(cls.REQUIRES_PYTHON)

    @classmethod
    def _is_missing(cls, requirement: str) -> bool:
        from ..utils.platform import find_executable

        if requirement in cls.REQUIRES_EXTERNAL:
            return find_executable(requirement) is None

        import importlib.util

        try:
            return importlib.util.find_spec(requirement) is None
        except (ImportError, ValueError):
            return True

    @classmethod
    def missing_dependencies(cls) -> List[str]:
        """Return every unmet requirement across all of this converter's pairs."""
        return [
            requirement
            for requirement in list(cls.REQUIRES_EXTERNAL) + list(cls.REQUIRES_PYTHON)
            if cls._is_missing(requirement)
        ]

    @classmethod
    def missing_for(cls, source_format: str, target_format: str) -> List[str]:
        """Return the unmet requirements for one specific format pair."""
        return [
            requirement
            for requirement in cls.requirements_for(source_format, target_format)
            if cls._is_missing(requirement)
        ]

    @classmethod
    def check_dependencies(cls) -> bool:
        """Return ``True`` when every declared requirement is satisfied."""
        return not cls.missing_dependencies()


class ConversionError(Exception):
    """Raised when a conversion fails."""

    def __init__(
        self,
        message: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        details: Optional[str] = None,
    ):
        self.source_format = source_format
        self.target_format = target_format
        self.details = details
        super().__init__(message)

    def __str__(self):
        msg = super().__str__()
        if self.source_format and self.target_format:
            msg = f"{msg} ({self.source_format} -> {self.target_format})"
        return msg


class DependencyError(ConversionError):
    """Raised when an optional dependency is missing."""

    def __init__(self, package: str, feature: str):
        self.package = package
        self.feature = feature
        super().__init__(
            f"Feature '{feature}' requires '{package}'. "
            f"Install with: pip install universal-converter[{feature}]"
        )


# --- Lazy re-export of concrete converters -------------------------------
# name -> submodule that defines it. Kept explicit so a typo raises
# AttributeError rather than silently importing the wrong module.
_CONVERTER_MODULES = {
    "AIConverter": "ai",
    "AudioConverter": "audio",
    "BioinformaticsConverter": "bioinformatics",
    "CloudConverter": "cloud",
    "DataConverter": "data",
    "DatabaseConverter": "database",
    "DocumentConverter": "documents",
    "GISConverter": "gis",
    "ImageConverter": "images",
    "NetworkConverter": "network",
    "VideoConverter": "video",
}


def __getattr__(name: str):
    module_name = _CONVERTER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals()) + list(_CONVERTER_MODULES))


__all__ = [
    "BaseConverter",
    "ConversionError",
    "ConversionResult",
    "ConversionTask",
    "DependencyError",
    *sorted(_CONVERTER_MODULES),
]
