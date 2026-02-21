#!/usr/bin/env python3
"""
Universal Converter - Convert Anything to Anything
The most comprehensive Python conversion library.

Installation:
    pip install universal-converter                   # Core
    pip install universal-converter[images]            # Image conversions
    pip install universal-converter[audio]            # Audio conversions
    pip install universal-converter[video]           # Video conversions
    pip install universal-converter[all]             # All features
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

from .converters import ConversionError, ConversionResult, DependencyError
from .utils import find_executable, run_command, get_platform_info


def __getattr__(name: str):
    """Lazy load converters"""
    if name == "convert_file" or name == "convert":
        from .converters.data import DataConverter
        def convert_file(path: str, output: str = None, from_fmt: str = None, to_fmt: str = None):
            from pathlib import Path
            from .converters import ConversionTask
            task = ConversionTask(
                source_path=Path(path),
                target_path=Path(output or path.replace(Path(path).suffix, f".{to_fmt or 'json'}")),
                source_format=from_fmt or Path(path).suffix[1:],
                target_format=to_fmt or Path(output or path).suffix[1:]
            )
            result = DataConverter().convert(task)
            if result.success:
                return result.output_path
            raise ConversionError(result.error)
        return convert_file
    
    if name == "resize_image":
        def resize_image(path: str, output: str = None, width: int = None, height: int = None, scale: float = None):
            from .converters.images import ImageConverter
            result = ImageConverter().resize(path, output, width, height, scale)
            if result.success:
                return result.output_path
            raise ConversionError(result.error)
        return resize_image
    
    # Placeholder for future converters - raises helpful error
    _future_converters = {"resize_image", "convert_audio", "convert_video", "convert_document"}
    if name in _future_converters:
        def _not_available(name):
            def _stub(*args, **kwargs):
                raise DependencyError(f"universal-converter[{name}]", name)
            return _stub
        return _not_available(name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConversionError",
    "ConversionResult", 
    "DependencyError",
    "convert",
    "convert_file",
    "resize_image",
    "find_executable",
    "run_command",
    "get_platform_info",
]


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Converter')
    parser.add_argument('input', nargs='?', default=None, help='Input file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-t', '--to', dest='to_fmt', help='Target format')
    parser.add_argument('-l', '--list', action='store_true', help='List supported formats')
    
    args = parser.parse_args()
    
    if args.list:
        print("Universal Converter supports:")
        print("  - Data: json, csv, xml, yaml, txt")
        print("  - Images: png, jpg, gif, bmp, tiff, webp")
        print("  - Documents: pdf, docx, md, html")
        print("  Install extras for more: pip install universal-converter[all]")
        return
    
    if args.input:
        convert_func = __getattr__('convert_file')
        result = convert_func(args.input, args.output, to_fmt=args.to_fmt)
        print(f"Converted: {result}")


if __name__ == "__main__":
    main()
