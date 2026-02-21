"""Base converter class and interfaces"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, ClassVar


@dataclass
class ConversionTask:
    """Immutable conversion task"""
    source_path: Path
    target_path: Path
    source_format: str
    target_format: str
    options: Dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: __import__('uuid').uuid4())


@dataclass
class ConversionResult:
    """Result of a conversion operation"""
    success: bool
    data: Any = None
    output_path: str = None
    error: str = None
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseConverter(ABC):
    """Base class for all converters"""
    
    SUPPORTED_CONVERSIONS: ClassVar[Dict[str, List[str]]] = {}
    PRIORITY: ClassVar[int] = 50
    REQUIRES_EXTERNAL: ClassVar[List[str]] = []
    
    @abstractmethod
    def convert(self, task: ConversionTask) -> ConversionResult:
        """Perform the conversion"""
        pass
    
    def can_handle(self, source_fmt: str, target_fmt: str) -> bool:
        """Check if this converter can handle the format pair"""
        return target_fmt in self.SUPPORTED_CONVERSIONS.get(source_fmt, [])
    
    def check_dependencies(self) -> bool:
        """Verify external tools are available"""
        from ..utils.platform import find_executable
        return all(find_executable(tool) for tool in self.REQUIRES_EXTERNAL)


class ConversionError(Exception):
    """Custom exception for conversion errors"""
    def __init__(self, message: str, source_format: str = None, target_format: str = None, details: str = None):
        self.source_format = source_format
        self.target_format = target_format
        self.details = details
        super().__init__(message)
    
    def __str__(self):
        msg = super().__str__()
        if self.source_format and self.target_format:
            msg = f"{msg} ({self.source_format} → {self.target_format})"
        return msg


class DependencyError(ConversionError):
    """Raised when an optional dependency is missing"""
    def __init__(self, package: str, feature: str):
        self.package = package
        self.feature = feature
        super().__init__(
            f"Feature '{feature}' requires '{package}'. Install with: pip install universal-converter[{feature}]"
        )
