#!/usr/bin/env python3
"""Universal Converter - registry-driven file format conversion.

Every conversion goes through the registry: the source and target formats are
resolved to a converter (or a chain of them), never to a hardcoded class.

    from universal_converter import convert_file, can_convert

    convert_file('data.json', 'report.html')     # direct
    convert_file('notes.md', 'notes.txt')        # direct
    convert_file('data.json', 'data.md')         # routed: json -> html -> md
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

__version__ = "1.1.0"

from .converters import (
    BaseConverter,
    ConversionError,
    ConversionResult,
    ConversionTask,
    DependencyError,
)
from .registry import (
    ConverterRegistry,
    can_convert,
    detect_format,
    find_converter,
    find_route,
    format_route,
    get_registry,
    list_conversions,
)
from .utils import find_executable, get_platform_info, run_command

#: Importable module / external binary -> how to obtain it.
INSTALL_HINTS: Dict[str, str] = {
    "PIL": "pip install universal-converter[images]",
    "docx": "pip install universal-converter[docx]",
    "openpyxl": "pip install universal-converter[xlsx]",
    "pypdf": "pip install universal-converter[pdf]",
    "reportlab": "pip install universal-converter[pdf]",
    "yaml": "pip install universal-converter[data]",
    "tomli_w": "pip install universal-converter[toml]",
    "torch": "pip install universal-converter[torch]",
    "onnx": "pip install universal-converter[torch]",
    "ffmpeg": "system package: apt install ffmpeg / brew install ffmpeg",
}


def convert_file(
    path: str,
    output: Optional[str] = None,
    from_fmt: Optional[str] = None,
    to_fmt: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    allow_multi_hop: bool = True,
    max_hops: int = 3,
) -> str:
    """Convert ``path`` and return the path that was written.

    The converter is chosen by the registry from the source and target
    formats. When no single converter declares the pair, a multi-hop route is
    searched (``json -> html -> md``); pass ``allow_multi_hop=False`` to
    require a direct conversion.

    Raises :class:`ConversionError` if no route exists or a step fails.
    """
    from .pipeline import convert_path

    result = convert_path(
        path,
        output=output,
        source_format=from_fmt,
        target_format=to_fmt,
        options=options,
        max_hops=max_hops,
        allow_multi_hop=allow_multi_hop,
    )
    if not result.success:
        raise ConversionError(result.error or "conversion failed", from_fmt, to_fmt)
    return result.output_path


#: Backwards-compatible alias.
convert = convert_file


def resize_image(
    path: str,
    output: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
) -> str:
    """Resize an image and return the output path.

    Defaults to ``<name>_resized<ext>`` next to the source; it never writes
    over the input file.
    """
    from .converters.images import ImageConverter

    result = ImageConverter().resize(path, output, width, height, scale)
    if result.success:
        return result.output_path
    raise ConversionError(result.error)


def thumbnail(path: str, output: Optional[str] = None, max_size=(128, 128)) -> str:
    """Write an aspect-preserving thumbnail and return the output path."""
    from .converters.images import ImageConverter

    result = ImageConverter().thumbnail(path, output, max_size)
    if result.success:
        return result.output_path
    raise ConversionError(result.error)


def plan_route(source_format: str, target_format: str, max_hops: int = 3):
    """Return the route the registry would take for a format pair."""
    from .pipeline import plan

    return plan(source_format, target_format, max_hops=max_hops, require_available=False)


__all__ = [
    "BaseConverter",
    "ConversionError",
    "ConversionResult",
    "ConversionTask",
    "ConverterRegistry",
    "DependencyError",
    "INSTALL_HINTS",
    "__version__",
    "can_convert",
    "convert",
    "convert_file",
    "detect_format",
    "find_converter",
    "find_executable",
    "find_route",
    "format_route",
    "get_platform_info",
    "get_registry",
    "list_conversions",
    "main",
    "plan_route",
    "resize_image",
    "run_command",
    "thumbnail",
]


# --- command line ---------------------------------------------------------


class _Style:
    """Minimal ANSI styling, disabled unless the stream is a terminal."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self(text, "1")

    def dim(self, text: str) -> str:
        return self(text, "2")

    def cyan(self, text: str) -> str:
        return self(text, "36")

    def green(self, text: str) -> str:
        return self(text, "32")

    def yellow(self, text: str) -> str:
        return self(text, "33")

    def red(self, text: str) -> str:
        return self(text, "31")


def _wrap(items: List[str], width: int, indent: str) -> List[str]:
    lines, current = [], ""
    for item in items:
        candidate = f"{current}, {item}" if current else item
        if len(candidate) > width and current:
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return [lines[0]] + [indent + line for line in lines[1:]] if lines else []


def _print_list(style: _Style) -> None:
    registry = get_registry()
    conversions = registry.supported_conversions()
    if not conversions:
        print("No converters discovered.")
        return

    total = registry.pair_count()
    routed = registry.routed_pair_count()
    print(
        style.bold(f"universal-converter {__version__}")
        + style.dim(
            f"  {total} direct pairs + {routed} routed"
            f" | {len(registry.formats())} formats"
            f" | {len(registry.converters)} converters"
        )
    )
    print()

    key_width = max(len(k) for k in conversions) + 2
    for source, targets in conversions.items():
        indent = " " * (key_width + 4)
        wrapped = _wrap(targets, 62, indent)
        head = f"  {style.cyan(source.rjust(key_width))} {style.dim('->')} {wrapped[0]}"
        print(head)
        for line in wrapped[1:]:
            print(line)

    print()
    print(
        style.dim(
            "Chained routes reach further: `--route SRC TGT` plans a path, "
            "and conversion uses one automatically."
        )
    )


def _print_doctor(style: _Style) -> int:
    registry = get_registry()
    conversions = registry.supported_conversions()

    print(style.bold(f"universal-converter {__version__} doctor"))
    print(style.dim(f"  python {sys.version.split()[0]} on {sys.platform}"))
    print()

    ready_total = blocked_total = 0
    blockers: Dict[str, int] = {}
    rows = []

    for converter in registry.converters:
        owned = [
            (source, target)
            for source, targets in conversions.items()
            for target in targets
            if registry.find_converter(source, target) is converter
        ]
        ready = 0
        converter_blockers: Dict[str, int] = {}
        for source, target in owned:
            missing = converter.missing_for(source, target)
            if missing:
                blocked_total += 1
                for name in missing:
                    blockers[name] = blockers.get(name, 0) + 1
                    converter_blockers[name] = converter_blockers.get(name, 0) + 1
            else:
                ready += 1
        ready_total += ready
        rows.append((converter, len(owned), ready, converter_blockers))

    if not rows:
        print(style.red("  no converters were discovered"))
        for module, reason in registry.import_errors.items():
            print(f"    {module}: {reason}")
        return 1

    name_width = max(len(c.__name__) for c, *_ in rows)
    print(
        style.dim(
            f"  {'converter'.ljust(name_width)}  prio  pairs  ready  status"
        )
    )
    for converter, owned, ready, converter_blockers in rows:
        if not converter_blockers:
            status = style.green("all dependencies present")
        else:
            detail = ", ".join(
                f"{name} (x{count})" for name, count in sorted(converter_blockers.items())
            )
            status = style.yellow(f"missing {detail}")
        print(
            f"  {converter.__name__.ljust(name_width)}  "
            f"{converter.PRIORITY:>4}  {owned:>5}  {ready:>5}  {status}"
        )

    for module, reason in registry.import_errors.items():
        print(f"  {style.red('import failed')} {module}: {reason}")

    print()
    total = ready_total + blocked_total
    summary = f"{ready_total}/{total} conversion pairs are runnable in this environment"
    print("  " + (style.green(summary) if not blockers else style.yellow(summary)))

    if blockers:
        print()
        print(style.dim("  install to unlock:"))
        for name, count in sorted(blockers.items(), key=lambda kv: -kv[1]):
            hint = INSTALL_HINTS.get(name, f"pip install {name}")
            print(f"    {style.bold(name):<24} {count:>3} pairs   {style.dim(hint)}")
    return 0


def _print_route(style: _Style, source: str, target: str, max_hops: int) -> int:
    from .pipeline import plan

    registry = get_registry()
    src, tgt = registry.normalize(source), registry.normalize(target)
    route = plan(src, tgt, max_hops=max_hops, require_available=False)
    if not route:
        print(
            style.red(f"no route from {src} to {tgt} within {max_hops} hops"),
            file=sys.stderr,
        )
        return 1

    label = "direct" if len(route) == 1 else f"{len(route)} hops"
    print(f"{style.bold(format_route(route))}  {style.dim(f'({label})')}")
    for index, step in enumerate(route, start=1):
        missing = step.converter.missing_for(step.source, step.target)
        mark = style.green("ok") if not missing else style.yellow(
            "needs " + ", ".join(missing)
        )
        print(
            f"  {index}. {step.source:>8} {style.dim('->')} {step.target:<8} "
            f"{style.cyan(step.converter.__name__):<28} {mark}"
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="universal-convert",
        description="Convert files between formats, chaining converters when needed.",
        epilog=(
            "examples:\n"
            "  universal-convert data.json -t csv\n"
            "  universal-convert data.json -o report.html\n"
            "  universal-convert README.md -t pdf\n"
            "  universal-convert --route json md\n"
            "  universal-convert --doctor\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input', nargs='?', help='input file')
    parser.add_argument('-o', '--output', help='output file (format inferred from it)')
    parser.add_argument('-t', '--to', dest='to_fmt', help='target format')
    parser.add_argument(
        '-f', '--from', dest='from_fmt', help='override the detected source format'
    )
    parser.add_argument(
        '-l', '--list', action='store_true', help='list supported conversions'
    )
    parser.add_argument(
        '--doctor',
        action='store_true',
        help='audit which conversions actually work in this environment',
    )
    parser.add_argument(
        '--route',
        nargs=2,
        metavar=('SRC', 'TGT'),
        help='show the conversion route between two formats and exit',
    )
    parser.add_argument(
        '--max-hops', type=int, default=3, help='maximum chained conversions (default 3)'
    )
    parser.add_argument(
        '--direct',
        action='store_true',
        help='refuse multi-hop routes; require a single converter',
    )
    parser.add_argument(
        '--color', choices=('auto', 'always', 'never'), default='auto',
        help='colourise output (default auto)',
    )
    parser.add_argument('-q', '--quiet', action='store_true', help='only print errors')
    parser.add_argument('-V', '--version', action='version', version=__version__)

    args = parser.parse_args(argv)
    style = _Style(
        args.color == 'always'
        or (args.color == 'auto' and sys.stdout.isatty())
    )

    if args.list:
        _print_list(style)
        return 0
    if args.doctor:
        return _print_doctor(style)
    if args.route:
        return _print_route(style, args.route[0], args.route[1], args.max_hops)
    if not args.input:
        parser.print_help()
        return 2

    from .pipeline import convert_path

    result = convert_path(
        args.input,
        output=args.output,
        source_format=args.from_fmt,
        target_format=args.to_fmt,
        max_hops=args.max_hops,
        allow_multi_hop=not args.direct,
    )

    if not result.success:
        print(style.red(f"error: {result.error}"), file=sys.stderr)
        return 1

    if not args.quiet:
        detail = (
            style.dim(f"  via {result.route_str}") if result.hops > 1 else ""
        )
        print(f"{style.green('converted')} {result.output_path}{detail}")
        for step_result in result.results:
            for warning in step_result.warnings:
                print(style.yellow(f"  warning: {warning}"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
